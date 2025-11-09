import pandas as pd
import numpy as np
import os
import json

def create_paper_dataset():
    """extract key data points from research paper and convert to structured dataframe"""

    # load data from JSON files in dataset folder
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')

    # load study data
    studies = []
    for study_file in ['study_1.json', 'study_2.json', 'study_3.json', 'study_4.json']:
        with open(os.path.join(dataset_dir, study_file), 'r') as f:
            studies.append(json.load(f))

    # load literature data
    with open(os.path.join(dataset_dir, 'literature_data.json'), 'r') as f:
        literature_data = json.load(f)

    # load mutation characteristics
    with open(os.path.join(dataset_dir, 'mutation_characteristics.json'), 'r') as f:
        mutation_characteristics = json.load(f)

    # combine into paper_data structure
    paper_data = {
        "studies": studies,
        "literature_data": literature_data,
        "mutation_characteristics": mutation_characteristics
    }

    # combine patient data from all studies
    all_patients = []
    source_id_counter = 0
    
    for study in paper_data["studies"]:
        study_id = study["study_id"]
        for patient in study["patient_data"]:
            patient_copy = patient.copy()
            patient_copy['study_id'] = study_id
            patient_copy['source_id'] = source_id_counter
            all_patients.append(patient_copy)
            source_id_counter += 1
    
    # create structured dataframe from combined patient data
    df = pd.DataFrame(all_patients)
    
    # remove duplicates based on patient_id
    print(f"Before deduplication: {len(df)} patients")
    df = df.drop_duplicates(subset=['patient_id'], keep='first')
    print(f"After deduplication: {len(df)} patients")

    # feature engineering

    # Extract RET variant from ret_variant field
    # Studies 1-3 patients are all K666N carriers (variant not explicitly stored in JSON)
    # Study 4 has explicit variant field for each patient
    df['ret_variant'] = df['ret_variant'].fillna('K666N')

    # Create RET risk level mapping based on ATA guidelines
    # Level 1 (Moderate): L790F, Y791F, V804M, S891A, K666N
    # Level 2 (High): C618S, C630R, C620Y
    # Level 3 (Highest): C634R, C634Y, C634W
    ret_risk_mapping = {
        'K666N': 1,
        'L790F': 1,
        'Y791F': 1,
        'V804M': 1,
        'S891A': 1,
        'C618S': 2,
        'C630R': 2,
        'C620Y': 2,
        'C634R': 3,
        'C634Y': 3,
        'C634W': 3
    }
    df['ret_risk_level'] = df['ret_variant'].map(ret_risk_mapping).fillna(1).astype(int)
    
    # convert calcitonin levels to numeric
    # Handle Study 4's numeric format (calcitonin_preoperative_basal) and Studies 1-3's string format
    def extract_calcitonin_numeric(row):
        # First, check if calcitonin_preoperative_basal exists (Study 4)
        if 'calcitonin_preoperative_basal' in row and pd.notna(row.get('calcitonin_preoperative_basal')):
            return float(row['calcitonin_preoperative_basal'])

        # Otherwise, extract from calcitonin_level string (Studies 1-3)
        calcitonin_str = str(row.get('calcitonin_level', ''))

        # Handle special cases
        if 'undetectable' in calcitonin_str.lower() or 'k/k' in calcitonin_str.lower():
            return 0.2  # use minimal detectable value

        # Extract first numeric value from string (handles "23/100" format by taking basal value)
        import re
        match = re.search(r'(\d+\.?\d*)', calcitonin_str)
        if match:
            return float(match.group(1))

        return 0.0

    df['calcitonin_level_numeric'] = df.apply(extract_calcitonin_numeric, axis=1)
    
    # calcitonin elevated flag - handle different normal ranges
    def determine_calcitonin_elevated(row):
        calcitonin_level = str(row.get('calcitonin_level', '')).lower()
        calcitonin_numeric = row['calcitonin_level_numeric']
        gender = row.get('gender', 0)  # 0=Female, 1=Male

        # handle special cases
        if 'undetectable' in calcitonin_level or 'normal' in calcitonin_level:
            return 0
        if 'not screened' in calcitonin_level or 'unknown' in calcitonin_level or 'not evaluated' in calcitonin_level:
            return 0

        # extract normal range from calcitonin_normal_range
        normal_range = str(row.get('calcitonin_normal_range', ''))

        # Study 4 uses gender-specific ranges: Males <10.3, Females <4.3
        if 'Males <10.3' in normal_range or 'Females <4.3' in normal_range:
            threshold = 10.3 if gender == 1 else 4.3
            return 1 if calcitonin_numeric > threshold else 0
        # Studies 1-3 ranges
        elif '0-5.1' in normal_range or '0.0-5.1' in normal_range:
            return 1 if calcitonin_numeric > 5.1 else 0
        elif '0-7.5' in normal_range or '0.0-7.5' in normal_range:
            return 1 if calcitonin_numeric > 7.5 else 0
        else:
            # default to 7.5 if range unclear
            return 1 if calcitonin_numeric > 7.5 else 0

    df['calcitonin_elevated'] = df.apply(determine_calcitonin_elevated, axis=1)
    
    # gender encoding (0=female, 1=male)
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
    
    # thyroid nodules
    df['thyroid_nodules_present'] = df['thyroid_ultrasound'].str.contains('nodules', case=False, na=False).astype(int)
    df['multiple_nodules'] = df['thyroid_nodule_count'].fillna(0) > 1
    df['multiple_nodules'] = df['multiple_nodules'].astype(int)
    
    # family history of MTC (based on relationship, family screening, and explicit family_history_mtc field)
    relationship_family = df['relationship'].isin(['Sister', 'Father', 'Paternal grandmother', 'Proband\'s daughter', 'Sister\'s son'])
    family_screening_yes = df['family_screening'].fillna('No').str.lower() == 'yes'
    explicit_family_history = df['family_history_mtc'].fillna('No').str.lower() == 'yes'
    
    df['family_history_mtc'] = (relationship_family | family_screening_yes | explicit_family_history).astype(int)
    
    # MTC diagnosis (target variable)
    df['mtc_diagnosis'] = df['mtc_diagnosis'].map({
        'No': 0, 
        'Yes': 1, 
        'Suspected (declined workup)': 0,
        'Suspected (elevated calcitonin)': 0,
        'Not screened': 0
    }).fillna(0).astype(int)
    
    # C-cell disease (includes MTC and C-cell hyperplasia/suspected)
    df['c_cell_disease'] = ((df['mtc_diagnosis'] == 1) | 
                           (df['c_cell_hyperplasia'] == 'Yes') | 
                           (df['c_cell_disease_suspected'] == 'Yes')).astype(int)
    
    # MEN2 syndrome (currently all No in paper data)
    df['men2_syndrome'] = df['men2_syndrome'].map({'No': 0, 'Yes': 1}).fillna(0).astype(int)
    
    # other clinical features
    df['pheochromocytoma'] = df['pheochromocytoma'].map({'No': 0, 'Yes': 1, 'Not screened': 0}).fillna(0).astype(int)
    df['hyperparathyroidism'] = df['hyperparathyroidism'].map({'No': 0, 'Yes': 1, 'Not screened': 0}).fillna(0).astype(int)
    
    # age groups for stratification
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 70, 100], labels=['young', 'middle', 'elderly', 'very_elderly'])
    
    # other flags (post-outcome/leaky features intentionally not included in model)
    
    # select final columns for ML
    final_columns = [
        'source_id', 'study_id', 'age', 'gender', 'ret_variant', 'ret_risk_level',
        'calcitonin_elevated', 'calcitonin_level_numeric',
        'thyroid_nodules_present', 'multiple_nodules', 'family_history_mtc', 'mtc_diagnosis',
        'c_cell_disease', 'men2_syndrome', 'pheochromocytoma', 'hyperparathyroidism', 'age_group'
    ]
    
    df_final = df[final_columns].copy()

    # save paper-only dataset
    df_final.to_csv('data/ret_multivariant_training_data.csv', index=False)

    # create expanded dataset with literature cases
    expanded_df = create_expanded_dataset(df_final, paper_data)

    # save expanded dataset
    expanded_df.to_csv('data/ret_multivariant_expanded_training_data.csv', index=False)
    
    return df_final, expanded_df

def create_expanded_dataset(paper_df, paper_data):
    """create expanded dataset with literature cases"""

    # literature MTC diagnosis ages
    lit_ages = paper_data['literature_data']['mtc_diagnosis_ages']

    # create synthetic cases based on literature
    np.random.seed(42)  # for reproducibility

    # Get variant distribution from paper_df to create realistic synthetic cases
    variant_distribution = paper_df['ret_variant'].value_counts(normalize=True).to_dict()
    risk_level_distribution = paper_df['ret_risk_level'].value_counts(normalize=True).to_dict()

    synthetic_cases = []
    for age in lit_ages:
        # Sample variant based on distribution in real data
        ret_variant = np.random.choice(list(variant_distribution.keys()),
                                       p=list(variant_distribution.values()))
        ret_risk_level = paper_df[paper_df['ret_variant'] == ret_variant]['ret_risk_level'].iloc[0]

        # create case with MTC
        case = {
            'age': age,
            'gender': np.random.choice([0, 1]),  # random gender
            'ret_variant': ret_variant,
            'ret_risk_level': ret_risk_level,
            'calcitonin_elevated': 1,
            'calcitonin_level_numeric': np.random.uniform(15, 60),  # elevated calcitonin
            'thyroid_nodules_present': 1,
            'multiple_nodules': np.random.choice([0, 1]),
            'family_history_mtc': 1,
            'mtc_diagnosis': 1,
            'c_cell_disease': 1,
            'men2_syndrome': 0,  # rare in heterozygous carriers
            'pheochromocytoma': 0,
            'hyperparathyroidism': 0,
            'age_group': 'middle' if 30 <= age <= 50 else 'elderly' if age > 50 else 'young',
            'declined_evaluation': 0,
            'underwent_surgery': 1
        }
        synthetic_cases.append(case)

    # create additional control cases (non-MTC carriers)
    control_cases = []
    for i in range(len(lit_ages) * 2):  # 2:1 control to case ratio
        age = np.random.choice(paper_df['age'].tolist() + lit_ages)

        # Sample variant based on distribution
        ret_variant = np.random.choice(list(variant_distribution.keys()),
                                       p=list(variant_distribution.values()))
        ret_risk_level = paper_df[paper_df['ret_variant'] == ret_variant]['ret_risk_level'].iloc[0]

        control = {
            'age': age,
            'gender': np.random.choice([0, 1]),
            'ret_variant': ret_variant,
            'ret_risk_level': ret_risk_level,
            'calcitonin_elevated': 0,
            'calcitonin_level_numeric': np.random.uniform(0, 7.5),  # normal calcitonin
            'thyroid_nodules_present': np.random.choice([0, 1], p=[0.7, 0.3]),  # lower nodule rate
            'multiple_nodules': 0,  # controls less likely to have multiple nodules
            'family_history_mtc': np.random.choice([0, 1], p=[0.6, 0.4]),
            'mtc_diagnosis': 0,
            'c_cell_disease': 0,
            'men2_syndrome': 0,
            'pheochromocytoma': 0,
            'hyperparathyroidism': 0,
            'age_group': 'middle' if 30 <= age <= 50 else 'elderly' if age > 50 else 'young',
            'declined_evaluation': 0,
            'underwent_surgery': 0
        }
        control_cases.append(control)
    
    # combine all cases
    expanded_df = pd.concat([paper_df, pd.DataFrame(synthetic_cases), pd.DataFrame(control_cases)], ignore_index=True)

    # remove duplicates based on key features (age, gender, calcitonin_level_numeric, ret_variant)
    print(f"Before expanded dataset deduplication: {len(expanded_df)} patients")
    expanded_df = expanded_df.drop_duplicates(subset=['age', 'gender', 'calcitonin_level_numeric', 'ret_variant'], keep='first')
    print(f"After expanded dataset deduplication: {len(expanded_df)} patients")

    return expanded_df

def print_dataset_info(df1, df2):
    """print dataset shapes and target distribution summaries"""
    print("=" * 60)
    print("DATASET CREATION SUMMARY")
    print("=" * 60)
    print(f"paper-only dataset shape: {df1.shape}")
    print(f"expanded dataset shape: {df2.shape}")
    print()
    
    # study breakdown
    print("STUDY BREAKDOWN:")
    study_counts = df1['study_id'].value_counts()
    for study_id, count in study_counts.items():
        if study_id == "study_1":
            study_name = "JCEM Case Reports (2025)"
        elif study_id == "study_2":
            study_name = "EDM Case Reports (2024)"
        elif study_id == "study_3":
            study_name = "Thyroid Journal (2016)"
        elif study_id == "study_4":
            study_name = "European Journal of Endocrinology (2006)"
        else:
            study_name = study_id
        print(f"- {study_name}: {count} patients")
    print()

    # variant breakdown
    print("RET VARIANT BREAKDOWN:")
    variant_counts = df1['ret_variant'].value_counts()
    for variant, count in variant_counts.items():
        risk_level = df1[df1['ret_variant'] == variant]['ret_risk_level'].iloc[0]
        risk_label = {1: 'Moderate', 2: 'High', 3: 'Highest'}[risk_level]
        print(f"- {variant} (Risk Level {risk_level} - {risk_label}): {count} patients")
    print()
    
    print("PAPER-ONLY DATASET TARGET DISTRIBUTION:")
    print(f"MTC diagnosis cases: {df1['mtc_diagnosis'].sum()}/{len(df1)} ({df1['mtc_diagnosis'].mean():.1%})")
    print(f"C-cell disease cases: {df1['c_cell_disease'].sum()}/{len(df1)} ({df1['c_cell_disease'].mean():.1%})")
    print(f"MEN2 syndrome cases: {df1['men2_syndrome'].sum()}/{len(df1)} ({df1['men2_syndrome'].mean():.1%})")
    print(f"Pheochromocytoma cases: {df1['pheochromocytoma'].sum()}/{len(df1)} ({df1['pheochromocytoma'].mean():.1%})")
    print(f"Hyperparathyroidism cases: {df1['hyperparathyroidism'].sum()}/{len(df1)} ({df1['hyperparathyroidism'].mean():.1%})")
    print()
    
    print("EXPANDED DATASET TARGET DISTRIBUTION:")
    print(f"MTC diagnosis cases: {df2['mtc_diagnosis'].sum()}/{len(df2)} ({df2['mtc_diagnosis'].mean():.1%})")
    print(f"C-cell disease cases: {df2['c_cell_disease'].sum()}/{len(df2)} ({df2['c_cell_disease'].mean():.1%})")
    print(f"MEN2 syndrome cases: {df2['men2_syndrome'].sum()}/{len(df2)} ({df2['men2_syndrome'].mean():.1%})")
    print()
    
    print("KEY FEATURES SUMMARY:")
    print(f"Average age: {df2['age'].mean():.1f} years")
    print(f"Gender distribution (M/F): {df2[df2['gender']==1]['gender'].count()}/{df2[df2['gender']==0]['gender'].count()}")
    print(f"Calcitonin elevated rate: {df2['calcitonin_elevated'].mean():.1%}")
    print(f"Thyroid nodules rate: {df2['thyroid_nodules_present'].mean():.1%}")
    print("=" * 60)

if __name__ == "__main__":
    paper_df, expanded_df = create_paper_dataset()
    print_dataset_info(paper_df, expanded_df)