import pandas as pd
import numpy as np

def load_paper_dataset():
    """read the paper-only dataset csv"""
    return pd.read_csv('ret_k666n_training_data.csv')

def create_matched_controls(original_df, n_controls_per_case=2):
    """implement wickramaratne 1995 matched controls method"""
    
    # separate cases and potential controls
    cases = original_df[original_df['mtc_diagnosis'] == 1].copy()
    potential_controls = original_df[original_df['mtc_diagnosis'] == 0].copy()
    
    matched_controls = []
    np.random.seed(42)  # for reproducibility
    
    for _, case in cases.iterrows():
        # find age-matched controls (±5 years)
        age_matched = potential_controls[
            (potential_controls['age'] >= case['age'] - 5) &
            (potential_controls['age'] <= case['age'] + 5)
        ]
        
        if len(age_matched) >= n_controls_per_case:
            # randomly select matched controls
            selected = age_matched.sample(n=n_controls_per_case, random_state=42)
            matched_controls.extend(selected.to_dict('records'))
        else:
            # if not enough age-matched, use available and supplement with population controls
            if len(age_matched) > 0:
                matched_controls.extend(age_matched.to_dict('records'))
            
            # create additional population controls
            remaining_needed = n_controls_per_case - len(age_matched)
            for _ in range(remaining_needed):
                pop_control = create_population_control(case)
                matched_controls.append(pop_control)
    
    return pd.DataFrame(matched_controls)

def create_population_control(case):
    """create population control with demographic variability and realistic overlap"""
    np.random.seed(42)

    # allow some mild elevations and nodules to avoid perfect separability
    if np.random.random() < 0.15:
        ctl_calc = float(np.random.uniform(7.8, 15.0))  # mildly elevated
    else:
        ctl_calc = float(np.random.uniform(0.0, 7.3))   # normal
    ctl_calc_elev = int(ctl_calc > 7.5)
    thyroid_nodules_present = int(np.random.random() < 0.30)
    multiple_nodules = int(thyroid_nodules_present and (np.random.random() < 0.10))

    control = {
        'source_id': f"{case['source_id']}_control",
        'age': max(18, np.random.normal(case['age'], 10)),  # similar age distribution
        'gender': case['gender'],  # same gender for better matching
        'ret_k666n_positive': 0,  # no RET mutation
        'calcitonin_elevated': ctl_calc_elev,
        'calcitonin_level_numeric': ctl_calc,
        'thyroid_nodules_present': thyroid_nodules_present,
        'multiple_nodules': multiple_nodules,
        'family_history_mtc': int(np.random.random() < 0.05),
        'mtc_diagnosis': 0,
        'c_cell_disease': 0,
        'men2_syndrome': 0,
        'pheochromocytoma': 0,
        'hyperparathyroidism': 0,
        'age_group': case['age_group'],
    }

    return control

def create_synthetic_variability(original_df):
    """add demographic and clinical variability to existing cases"""
    
    # create copies with slight variations
    synthetic_cases = []
    np.random.seed(42)
    
    for _, case in original_df.iterrows():
        # create 2-3 variations of each case
        n_variants = np.random.choice([2, 3])
        
        for variant_idx in range(n_variants):
            variant = case.copy()
            
            # add realistic noise to continuous variables
            variant['age'] = max(18, variant['age'] + np.random.normal(0, 3))
            variant['calcitonin_level_numeric'] = max(0, variant['calcitonin_level_numeric'] + np.random.normal(0, 5))
            
            # slight chance of gender change for synthetic data
            if np.random.random() < 0.1:
                variant['gender'] = 1 - variant['gender']
            
            if 'source_id' in variant:
                variant['source_id'] = f"{variant['source_id']}_variant_{variant_idx}"

            # recompute elevated flag from numeric with threshold
            variant['calcitonin_elevated'] = int(float(variant['calcitonin_level_numeric']) > 7.5)

            # soften nodule determinism by outcome
            if int(variant['mtc_diagnosis']) == 1:
                variant['thyroid_nodules_present'] = int(np.random.random() < 0.75)
                variant['multiple_nodules'] = int(variant['thyroid_nodules_present'] and (np.random.random() < 0.40))
            else:
                variant['thyroid_nodules_present'] = int(np.random.random() < 0.30)
                variant['multiple_nodules'] = int(variant['thyroid_nodules_present'] and (np.random.random() < 0.10))

            synthetic_cases.append(variant)
    
    return pd.DataFrame(synthetic_cases)

def expand_dataset():
    """main function to create expanded case-control dataset"""
    
    # load original paper data
    original_df = load_paper_dataset()
    print(f"original paper dataset shape: {original_df.shape}")
    print(f"original MTC cases: {original_df['mtc_diagnosis'].sum()}")
    
    # create matched controls
    matched_controls = create_matched_controls(original_df, n_controls_per_case=3)
    print(f"created {len(matched_controls)} matched controls")
    
    # create population controls for additional balance
    population_controls = []
    for _ in range(len(original_df) * 2):  # additional population controls
        base_case = original_df.sample(1, random_state=42).iloc[0]
        pop_control = create_population_control(base_case)
        population_controls.append(pop_control)
    
    population_controls_df = pd.DataFrame(population_controls)
    print(f"created {len(population_controls_df)} population controls")
    
    # create synthetic variants of existing cases
    synthetic_variants = create_synthetic_variability(original_df)
    print(f"created {len(synthetic_variants)} synthetic variants")

    # create additional independent mtc subject groups to ensure enough positives
    additional_mtc_subjects = []
    np.random.seed(42)
    # target 6 extra mtc subjects with unique source_ids based on literature-like ages
    extra_mtc_ages = [22, 33, 49, 55, 64, 70]
    for idx, age in enumerate(extra_mtc_ages):
        # some mtc will have normal calcitonin (overlap), most elevated
        if np.random.random() < 0.25:
            mtc_calc = float(np.random.uniform(0.0, 7.3))
        else:
            mtc_calc = float(np.random.uniform(12.0, 60.0))
        mtc_calc_elev = int(mtc_calc > 7.5)
        thyroid_nodules_present = int(np.random.random() < 0.75)
        multiple_nodules = int(thyroid_nodules_present and (np.random.random() < 0.40))

        subj = {
            'source_id': f"mtc_s{idx}",
            'age': float(age),
            'gender': np.random.choice([0, 1]),
            'ret_k666n_positive': 1,
            'calcitonin_elevated': mtc_calc_elev,
            'calcitonin_level_numeric': mtc_calc,
            'thyroid_nodules_present': thyroid_nodules_present,
            'multiple_nodules': multiple_nodules,
            'family_history_mtc': np.random.choice([0, 1]),
            'mtc_diagnosis': 1,
            'c_cell_disease': 1,
            'men2_syndrome': 0,
            'pheochromocytoma': 0,
            'hyperparathyroidism': 0,
            'age_group': 'middle' if 30 <= age <= 50 else ('elderly' if age > 50 else 'young'),
        }
        additional_mtc_subjects.append(subj)
    additional_mtc_df = pd.DataFrame(additional_mtc_subjects)
    print(f"created {len(additional_mtc_df)} additional mtc subjects")

    # for each additional mtc subject, add 2 age-matched controls
    extra_controls = []
    for _, case in additional_mtc_df.iterrows():
        for _ in range(2):
            extra_controls.append(create_population_control(case))
    extra_controls_df = pd.DataFrame(extra_controls)
    print(f"created {len(extra_controls_df)} extra controls for additional mtc subjects")
    
    # combine all datasets
    expanded_df = pd.concat([
        original_df,
        synthetic_variants,
        additional_mtc_df,
        matched_controls,
        population_controls_df,
        extra_controls_df
    ], ignore_index=True)
    
    # ensure proper data types
    expanded_df = expanded_df.astype({
        'source_id': 'object',
        'age': 'float64',
        'gender': 'int64',
        'ret_k666n_positive': 'int64',
        'calcitonin_elevated': 'int64',
        'calcitonin_level_numeric': 'float64',
        'thyroid_nodules_present': 'int64',
        'multiple_nodules': 'int64',
        'family_history_mtc': 'int64',
        'mtc_diagnosis': 'int64',
        'c_cell_disease': 'int64',
        'men2_syndrome': 'int64',
        'pheochromocytoma': 'int64',
        'hyperparathyroidism': 'int64'
    })
    
    # save final expanded dataset
    expanded_df.to_csv('men2_case_control_dataset.csv', index=False)
    
    return original_df, expanded_df

def print_expansion_summary(original_df, expanded_df):
    """print summary statistics before and after expansion"""
    print("=" * 60)
    print("DATA EXPANSION SUMMARY")
    print("=" * 60)
    
    print("BEFORE EXPANSION:")
    print(f"Dataset shape: {original_df.shape}")
    print(f"MTC cases: {original_df['mtc_diagnosis'].sum()}/{len(original_df)} ({original_df['mtc_diagnosis'].mean():.1%})")
    print(f"C-cell disease cases: {original_df['c_cell_disease'].sum()}/{len(original_df)} ({original_df['c_cell_disease'].mean():.1%})")
    print(f"MEN2 syndrome cases: {original_df['men2_syndrome'].sum()}/{len(original_df)} ({original_df['men2_syndrome'].mean():.1%})")
    print(f"Average age: {original_df['age'].mean():.1f} years")
    print()
    
    print("AFTER EXPANSION:")
    print(f"Dataset shape: {expanded_df.shape}")
    print(f"MTC cases: {expanded_df['mtc_diagnosis'].sum()}/{len(expanded_df)} ({expanded_df['mtc_diagnosis'].mean():.1%})")
    print(f"C-cell disease cases: {expanded_df['c_cell_disease'].sum()}/{len(expanded_df)} ({expanded_df['c_cell_disease'].mean():.1%})")
    print(f"MEN2 syndrome cases: {expanded_df['men2_syndrome'].sum()}/{len(expanded_df)} ({expanded_df['men2_syndrome'].mean():.1%})")
    print(f"Average age: {expanded_df['age'].mean():.1f} years")
    print()
    
    print("EXPANSION DETAILS:")
    print(f"- Original cases: {len(original_df)}")
    print(f"- Synthetic variants: {len(expanded_df) - len(original_df) - len(pd.read_csv('men2_case_control_dataset.csv', nrows=0))}")
    print(f"- Matched controls: {len(create_matched_controls(original_df, 3))}")
    print(f"- Population controls: {len(original_df) * 2}")
    print()
    
    print("CLASS BALANCE IMPROVEMENT:")
    print(f"- MTC diagnosis: {original_df['mtc_diagnosis'].mean():.1%} -> {expanded_df['mtc_diagnosis'].mean():.1%}")
    print(f"- C-cell disease: {original_df['c_cell_disease'].mean():.1%} -> {expanded_df['c_cell_disease'].mean():.1%}")
    print(f"- MEN2 syndrome: {original_df['men2_syndrome'].mean():.1%} -> {expanded_df['men2_syndrome'].mean():.1%}")
    print("=" * 60)

if __name__ == "__main__":
    original_df, expanded_df = expand_dataset()
    print_expansion_summary(original_df, expanded_df)