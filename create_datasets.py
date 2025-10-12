import pandas as pd
import numpy as np
import os

def create_paper_dataset():
    """extract key data points from research paper and convert to structured dataframe"""
    
    # paper data from research study
    paper_data = {
        "study_info": {
            "title": "Medullary Thyroid Carcinoma and Clinical Outcomes in Heterozygous Carriers of the RET K666N Germline Pathogenic Variant",
            "journal": "JCEM Case Reports",
            "publication_date": "March 2025",
            "study_type": "Case series"
        },

        "patient_data": [
            {
                "patient_id": "Patient_1_Proband",
                "age": 24,
                "gender": "Female",
                "relationship": "Proband",
                "ret_variant": "K666N",
                "ret_status": "Positive",
                "calcitonin_level": "Undetectable",
                "calcitonin_normal_range": "0.0-7.5 pg/mL",
                "thyroid_ultrasound": "Normal",
                "mtc_diagnosis": "No",
                "pheochromocytoma": "No",
                "hyperparathyroidism": "No",
                "men2_syndrome": "No",
                "genetic_testing_reason": "Breast cancer risk stratification"
            },
            {
                "patient_id": "Patient_2_Sister",
                "age": 21,
                "gender": "Female",
                "relationship": "Sister",
                "ret_variant": "K666N",
                "ret_status": "Positive",
                "calcitonin_level": "Undetectable",
                "calcitonin_normal_range": "0.0-7.5 pg/mL",
                "thyroid_ultrasound": "Normal",
                "mtc_diagnosis": "No",
                "pheochromocytoma": "No",
                "hyperparathyroidism": "No",
                "men2_syndrome": "No",
                "genetic_testing_reason": "Family screening"
            },
            {
                "patient_id": "Patient_3_Father",
                "age": 60,
                "gender": "Male",
                "relationship": "Father",
                "ret_variant": "K666N",
                "ret_status": "Positive",
                "calcitonin_level": "13 pg/mL",
                "calcitonin_normal_range": "0.0-7.5 pg/mL",
                "calcitonin_elevated": "Yes",
                "thyroid_ultrasound": "Multiple subcentimeter solid hypoechoic nodules",
                "thyroid_nodule_count": 3,
                "largest_nodule_size_mm": 6,
                "biopsy_result": "Bethesda V - Suspicious for malignancy",
                "mtc_diagnosis": "Yes",
                "mtc_stage": "pT1aNxMx",
                "mtc_size_mm": 8,
                "mtc_bilateral": "Yes",
                "c_cell_hyperplasia": "Yes",
                "treatment": "Total thyroidectomy",
                "postop_calcitonin": "Undetectable",
                "pheochromocytoma": "No",
                "hyperparathyroidism": "No",
                "men2_syndrome": "No",
                "family_screening": "Yes"
            },
            {
                "patient_id": "Patient_4_Grandmother",
                "age": 84,
                "gender": "Female",
                "relationship": "Paternal grandmother",
                "ret_variant": "K666N",
                "ret_status": "Positive",
                "calcitonin_level": "37 pg/mL",
                "calcitonin_normal_range": "0.0-7.5 pg/mL",
                "calcitonin_elevated": "Yes",
                "thyroid_ultrasound": "Multiple thyroid nodules",
                "mtc_diagnosis": "Suspected (declined workup)",
                "c_cell_disease_suspected": "Yes",
                "pheochromocytoma": "No",
                "hyperparathyroidism": "No",
                "men2_syndrome": "No",
                "declined_further_evaluation": "Yes"
            }
        ],

        "literature_data": {
            "ret_k666n_families_reported": 8,
            "ret_k666n_carriers_reported": 24,
            "mtc_cases_in_literature": 9,
            "c_cell_disease_cases": 2,
            "mtc_diagnosis_ages": [22, 23, 33, 49, 51, 55, 59, 64, 70],
            "penetrance": "Incomplete",
            "expressivity": "Age-dependent",
            "men2_features": "Rare in heterozygous carriers"
        },

        "mutation_characteristics": {
            "mutation_type": "Missense",
            "nucleotide_change": "c.1998G>T",
            "protein_change": "p.Lys666Asn",
            "exon": "14",
            "domain": "Intracellular juxtamembrane domain",
            "mechanism": "Ligand-independent kinase activation",
            "ata_risk_level": "Not yet assigned"
        }
    }

    # create structured dataframe from patient data
    df = pd.DataFrame(paper_data["patient_data"])
    
    # add stable source identifier per original subject
    df['source_id'] = range(len(df))
    
    # feature engineering
    df['ret_k666n_positive'] = 1  # all patients have the mutation
    
    # convert calcitonin levels to numeric
    df['calcitonin_level_numeric'] = df['calcitonin_level'].str.extract(r'(\d+\.?\d*)').iloc[:, 0].astype(float)
    df['calcitonin_level_numeric'] = df['calcitonin_level_numeric'].fillna(0.0)
    
    # calcitonin elevated flag
    contains_units = df['calcitonin_level'].str.contains('pg/mL', na=False)
    extracted_numeric = df['calcitonin_level'].str.extract(r'(\d+\.?\d*)').iloc[:, 0].astype(float)
    is_elevated_numeric = (extracted_numeric > 7.5).fillna(False)
    df['calcitonin_elevated'] = (contains_units & is_elevated_numeric).astype(int)
    
    # gender encoding (0=female, 1=male)
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
    
    # thyroid nodules
    df['thyroid_nodules_present'] = df['thyroid_ultrasound'].str.contains('nodules', case=False, na=False).astype(int)
    df['multiple_nodules'] = df['thyroid_nodule_count'].fillna(0) > 1
    df['multiple_nodules'] = df['multiple_nodules'].astype(int)
    
    # family history of MTC (based on relationship and family screening)
    df['family_history_mtc'] = df['relationship'].isin(['Sister', 'Father', 'Paternal grandmother']) | df['family_screening'].fillna('No').str.lower() == 'yes'
    df['family_history_mtc'] = df['family_history_mtc'].astype(int)
    
    # MTC diagnosis (target variable)
    df['mtc_diagnosis'] = df['mtc_diagnosis'].map({'No': 0, 'Yes': 1, 'Suspected (declined workup)': 0}).fillna(0).astype(int)
    
    # C-cell disease (includes MTC and C-cell hyperplasia/suspected)
    df['c_cell_disease'] = ((df['mtc_diagnosis'] == 1) | 
                           (df['c_cell_hyperplasia'] == 'Yes') | 
                           (df['c_cell_disease_suspected'] == 'Yes')).astype(int)
    
    # MEN2 syndrome (currently all No in paper data)
    df['men2_syndrome'] = df['men2_syndrome'].map({'No': 0, 'Yes': 1}).fillna(0).astype(int)
    
    # other clinical features
    df['pheochromocytoma'] = df['pheochromocytoma'].map({'No': 0, 'Yes': 1}).fillna(0).astype(int)
    df['hyperparathyroidism'] = df['hyperparathyroidism'].map({'No': 0, 'Yes': 1}).fillna(0).astype(int)
    
    # age groups for stratification
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 70, 100], labels=['young', 'middle', 'elderly', 'very_elderly'])
    
    # other flags (post-outcome/leaky features intentionally not included in model)
    
    # select final columns for ML
    final_columns = [
        'source_id', 'age', 'gender', 'ret_k666n_positive', 'calcitonin_elevated', 'calcitonin_level_numeric',
        'thyroid_nodules_present', 'multiple_nodules', 'family_history_mtc', 'mtc_diagnosis',
        'c_cell_disease', 'men2_syndrome', 'pheochromocytoma', 'hyperparathyroidism', 'age_group'
    ]
    
    df_final = df[final_columns].copy()
    
    # save paper-only dataset
    df_final.to_csv('data/ret_k666n_training_data.csv', index=False)
    
    # create expanded dataset with literature cases
    expanded_df = create_expanded_dataset(df_final, paper_data)
    
    # save expanded dataset  
    expanded_df.to_csv('data/ret_k666n_expanded_training_data.csv', index=False)
    
    return df_final, expanded_df

def create_expanded_dataset(paper_df, paper_data):
    """create expanded dataset with literature cases"""
    
    # literature MTC diagnosis ages
    lit_ages = paper_data['literature_data']['mtc_diagnosis_ages']
    
    # create synthetic cases based on literature
    np.random.seed(42)  # for reproducibility
    
    synthetic_cases = []
    for age in lit_ages:
        # create case with MTC
        case = {
            'age': age,
            'gender': np.random.choice([0, 1]),  # random gender
            'ret_k666n_positive': 1,
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
        control = {
            'age': age,
            'gender': np.random.choice([0, 1]),
            'ret_k666n_positive': 1,
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
    
    return expanded_df

def print_dataset_info(df1, df2):
    """print dataset shapes and target distribution summaries"""
    print("=" * 60)
    print("DATASET CREATION SUMMARY")
    print("=" * 60)
    print(f"paper-only dataset shape: {df1.shape}")
    print(f"expanded dataset shape: {df2.shape}")
    print()
    
    print("PAPER-ONLY DATASET TARGET DISTRIBUTION:")
    print(f"MTC diagnosis cases: {df1['mtc_diagnosis'].sum()}/{len(df1)} ({df1['mtc_diagnosis'].mean():.1%})")
    print(f"C-cell disease cases: {df1['c_cell_disease'].sum()}/{len(df1)} ({df1['c_cell_disease'].mean():.1%})")
    print(f"MEN2 syndrome cases: {df1['men2_syndrome'].sum()}/{len(df1)} ({df1['men2_syndrome'].mean():.1%})")
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