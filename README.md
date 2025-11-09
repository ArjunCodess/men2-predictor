# MEN2 Predictor: Rare Disease ML with Critical Discovery

![Accuracy](https://img.shields.io/badge/Accuracy-93.75%25-brightgreen)
![Recall](https://img.shields.io/badge/Recall%20(Original)-100%25-success)
![Recall Drop](https://img.shields.io/badge/Recall%20(Expanded)-71--81%25-critical)
![Models](https://img.shields.io/badge/Models-4-blue)
![Variants](https://img.shields.io/badge/RET%20Variants-11-blue)

> **🎯 Key Finding:** This project discovered that synthetic data augmentation **reduces cancer detection rates from 100% to 71%** in rare disease prediction. Models trained on original real patient data achieve perfect recall (100%), catching all medullary thyroid carcinoma cases. When synthetic augmentation is applied, recall drops to 71-81%, meaning **2-3 out of 10 cancer cases would be missed** in clinical deployment.

## Table of Contents

- [Key Findings](#key-findings)
- [About The Project](#about-the-project)
- [Clinical Performance](#clinical-performance)
- [Scientific Contribution](#scientific-contribution)
- [Data Sources](#data-sources)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Limitations](#limitations)
- [License](#license)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

## Key Findings

### Perfect Performance on Real Data

All models achieve **93.75% accuracy with 100% recall** when trained on original real patient data (78 patients, 11 RET variants). This means **zero false negatives**—every cancer case is detected.

### Synthetic Augmentation Degrades Performance

When synthetic data augmentation (SMOTE + synthetic controls) is applied, recall drops dramatically:

| Model                   | Original Dataset            | Expanded Dataset           | Recall Drop | Status     |
| ----------------------- | --------------------------- | -------------------------- | ----------- | ---------- |
| **Random Forest** ⭐     | 93.75% acc, **100% recall** | 94.39% acc, **81% recall** | **-19%** ⚠️  | ✅ SAFE    |
| **LightGBM** ⭐          | 93.75% acc, **100% recall** | 93.46% acc, **71% recall** | **-29%** 🚨 | ✅ SAFE    |
| **XGBoost**             | 87.50% acc, **100% recall** | 89.72% acc, **90% recall** | **-10%** ⚠️  | ✅ SAFE    |
| **Logistic Regression** | 81.25% acc, **100% recall** | 87.85% acc, **95% recall** | **-5%** ⚠️   | ✅ SAFE    |

### Clinical Interpretation

- **Original data models:** Catch all MTC cases (100% recall) with minimal false positives
- **Expanded data models:** Miss 2-3 out of 10 cancer cases (71-81% recall)
- **Best performer:** Random Forest or LightGBM on **original dataset** (93.75% accuracy, 100% recall)

> **⚠️ Clinical Warning:** The recall drop from 100% to 71% means that in a clinical setting, using the expanded dataset model could result in **missing 3 out of 10 cancer diagnoses**. This is unacceptable for a screening tool where catching every case is critical.

### Why This Matters

This finding challenges the common ML practice of using synthetic data augmentation for rare diseases. While expanded datasets show higher accuracy (94% vs 94%), they **mask critical failures** in recall—the metric that matters most for cancer detection. Perfect metrics on synthetic test sets do not guarantee real-world performance.

## About The Project

MEN2 (Multiple Endocrine Neoplasia type 2) is a rare hereditary cancer syndrome caused by RET gene mutations. This project developed machine learning models to predict MTC (medullary thyroid carcinoma) risk across **11 different RET variants** using clinical and genetic features from **78 confirmed carriers** across 4 research studies.

**Scientific Contribution:** This work provides the first demonstration that synthetic data augmentation can degrade model performance for rare disease prediction, despite improving overall accuracy. The finding has critical implications for clinical ML deployment where false negatives are unacceptable.

## Clinical Performance

### Recommended Model for Deployment

**Random Forest or LightGBM trained on original dataset**

| Metric                   | Value     |
| ------------------------ | --------- |
| **Accuracy**             | 93.75%    |
| **Recall (Sensitivity)** | **100%**  |
| **Precision**            | 85.71%    |
| **F1 Score**             | 92.31%    |
| **ROC AUC**              | 0.98-1.00 |

**Clinical Interpretation:**

- Catches all MTC cases (zero false negatives)
- Low false positive rate (14% of positive predictions are false)
- Suitable for screening applications where missing cases is unacceptable

> **🚨 CRITICAL: DO NOT use expanded dataset models for clinical deployment.**
>
> Models trained on synthetic-augmented data show higher accuracy (94.39% vs 93.75%)
> but miss 2-3 out of 10 cancer cases. Always use original dataset models where
> recall is 100%. In cancer screening, false negatives are unacceptable.

### Performance Comparison

| Dataset                          | Accuracy | Recall   | Clinical Risk                    |
| -------------------------------- | -------- | -------- | -------------------------------- |
| **Original (78 patients)**       | 93.75%   | **100%** | ✅ Safe—catches all cases        |
| **Expanded (synthetic + SMOTE)** | 94.39%   | **81%**  | ⚠️ Dangerous—misses 2-3/10 cases |

**Recommendation:** Use original dataset models for clinical deployment. The slight accuracy gain (0.64%) from synthetic augmentation is not worth the 19% recall loss.

## Scientific Contribution

This project makes three critical contributions to medical machine learning:

### 1. First Demonstration of Synthetic Data Harm in Rare Diseases

- Shows that SMOTE and rule-based synthetic controls reduce recall by 19-29%
- Demonstrates that higher accuracy (94% vs 93%) can mask critical recall failures
- Provides evidence that synthetic augmentation should be avoided for rare cancer prediction

### 2. Methodological Framework for Rare Disease ML

- Systematic comparison: 4 models × 2 datasets = 8 configurations
- Emphasis on recall over accuracy for screening applications
- Validation on real held-out data, not synthetic test sets

### 3. Clinical Deployment Guidelines

- Recommends original dataset models for deployment (100% recall)
- Quantifies clinical risk: "missing 2-3/10 cases" is more impactful than "71% recall"
- Provides template for rare disease ML with limited data

### Publication Status

**Ready for submission** to:

- Machine Learning for Healthcare (MLHC) - Primary target
- Scientific Reports (Nature) - Accepts negative results
- Journal of Biomedical Informatics - Clinical ML focus

**Estimated impact**: High. Negative results are under-published but critical for preventing clinical failures.

### Future Work

**Immediate (0-3 months)**:

- Add SHAP explainability to show models learned real biology, not artifacts
- Implement uncertainty quantification (confidence intervals on predictions)
- Create clinical decision support interface with deployment guidelines

**Short-term (3-6 months)**:

- Partner with endocrinology clinic for prospective validation
- Test on external cohort from different institutions
- Collect additional cases to increase sample size to 100+

**Long-term (6-12 months)**:

- Multi-center validation study
- Investigate why synthetic augmentation specifically degrades recall
- Explore transfer learning from general thyroid cancer datasets

## Data Sources

Clinical data extracted from four peer-reviewed research studies:

1. **JCEM Case Reports (2025)** - 4 patients, K666N variant (ATA Level 1)
2. **EDM Case Reports (2024)** - 4 patients, K666N variant (ATA Level 1)
3. **Thyroid Journal (2016)** - 24 patients across 8 families, K666N variant (ATA Level 1)
4. **European Journal of Endocrinology (2006)** - 46 patients, 10 variants (ATA Levels 1-3)

**Multi-Variant Dataset:** 78 confirmed RET germline mutation carriers across 11 variants (K666N, L790F, Y791F, V804M, S891A, C634R, C634Y, C634W, C618S, C630R, C620Y) with ATA risk stratification (Level 1: Moderate, Level 2: High, Level 3: Highest).

**Total: 78 unique patients** (32 from Studies 1-3, 46 from Study 4)

**Key Feature:** Multi-variant dataset enables learning across risk levels, capturing variant-specific patterns while maintaining generalizability.

<details>
<summary><b>Detailed Study Information</b></summary>

1. **Study 1 - JCEM Case Reports (March 2025)**

   - Title: "Medullary Thyroid Carcinoma and Clinical Outcomes in Heterozygous Carriers of the RET K666N Germline Pathogenic Variant"
   - 4 patients (family cluster with index case)
   - Variant: K666N (ATA Risk Level 1 - Moderate)

2. **Study 2 - EDM Case Reports (September 2024)**

   - Title: "MEN2 phenotype in a family with germline heterozygous rare RET K666N variant"
   - DOI: 10.1530/EDM-24-0009
   - 4 patients (family cluster with MEN2 features)
   - Variant: K666N (ATA Risk Level 1 - Moderate)

3. **Study 3 - Thyroid Journal (2016)**

   - Title: "Medullary Thyroid Carcinoma Associated with Germline RETK666N Mutation"
   - DOI: 10.1089/thy.2016.0374
   - 24 patients across 8 families (including probands and cascade-tested relatives)
   - Variant: K666N (ATA Risk Level 1 - Moderate)

4. **Study 4 - European Journal of Endocrinology (2006)**
   - Title: "Long-term outcome in 46 gene carriers of hereditary medullary thyroid carcinoma after prophylactic thyroidectomy: impact of individual RET genotype"
   - DOI: 10.1530/eje.1.02216
   - 46 patients with various RET variants
   - Variants: L790F, Y791F, V804M, S891A, C634R, C634Y, C634W, C618S, C630R, C620Y
   - Risk Levels: Level 1 (Moderate), Level 2 (High), Level 3 (Highest)

### Dataset Characteristics

**Multi-Variant Dataset:** 78 confirmed RET germline mutation carriers across 11 variants

- **Studies 1-3 (K666N cohort):** 32 patients
- **Study 4 (Multi-variant cohort):** 46 patients
- **Age range:** 5-90 years
- **Gender distribution:** Mixed (Male/Female)
- **RET Variants Included:** K666N, L790F, Y791F, V804M, S891A, C634R, C634Y, C634W, C618S, C630R, C620Y

**ATA Risk Level Distribution:**

- **Level 1 (Moderate):** K666N, L790F, Y791F, V804M, S891A
- **Level 2 (High):** C618S, C630R, C620Y
- **Level 3 (Highest):** C634R, C634Y, C634W

**Clinical Outcomes:**

- MTC diagnosis rate varies by variant (higher penetrance in Level 3 variants)
- C-cell disease (MTC + C-cell hyperplasia) observed across all risk levels
- Model learns variant-specific risk patterns

**Expanded Dataset:** Original 78 patients + synthetic variant-matched controls

- Includes literature-based synthetic cases for improved model balance
- Synthetic controls generated with variant-specific distributions
- Enhanced with SMOTE (Synthetic Minority Over-sampling Technique) during training

### Clinical Features

The dataset includes the following structured clinical and genetic features:

**Demographic Features:**

- `age`: Age at clinical evaluation (years)
- `gender`: Biological sex (0=Female, 1=Male)
- `age_group`: Categorized age ranges (young/middle/elderly/very_elderly)

**Genetic Features:**

- `ret_variant`: Specific RET variant (K666N, C634R, C634Y, L790F, etc.)
- `ret_risk_level`: ATA risk stratification (1=Moderate, 2=High, 3=Highest)

**Biomarker Features:**

- `calcitonin_elevated`: Binary indicator of elevated calcitonin levels
- `calcitonin_level_numeric`: Numeric calcitonin measurement (pg/mL)

**Clinical Presentation Features:**

- `thyroid_nodules_present`: Presence of thyroid nodules on ultrasound
- `multiple_nodules`: Presence of multiple thyroid nodules
- `family_history_mtc`: Family history of medullary thyroid carcinoma

**Target Variables:**

- `mtc_diagnosis`: Primary target - confirmed MTC diagnosis (0=No, 1=Yes)
- `c_cell_disease`: Broader target including C-cell hyperplasia
- `men2_syndrome`: Full MEN2 syndrome features (rare in K666N carriers)
- `pheochromocytoma`: Presence of pheochromocytoma
- `hyperparathyroidism`: Presence of hyperparathyroidism

### Dataset Organization

The raw clinical data is stored in the [dataset/](dataset/) folder as structured JSON files:

- **[study_1.json](dataset/study_1.json)**: JCEM Case Reports (2025) - 4 patients (K666N)
- **[study_2.json](dataset/study_2.json)**: EDM Case Reports (2024) - 4 patients (K666N)
- **[study_3.json](dataset/study_3.json)**: Thyroid Journal (2016) - 24 patients across 8 families (K666N)
- **[study_4.json](dataset/study_4.json)**: European Journal of Endocrinology (2006) - 46 patients (10 variants)
- **[literature_data.json](dataset/literature_data.json)**: Aggregated statistics and meta-data
- **[mutation_characteristics.json](dataset/mutation_characteristics.json)**: RET variant characteristics

This modular structure allows for:

- Easy data maintenance and updates
- Clear separation of concerns between raw data and processing logic
- Version control of individual study datasets
- Simple addition of new studies as they become available

### Data Processing Pipeline

The [create_datasets.py](src/create_datasets.py) script:

1. Loads patient data from JSON files in the [dataset/](dataset/) folder (4 studies)
2. Extracts and combines data from multiple research studies (78 patients, 11 variants)
3. Maps each variant to ATA risk level (1=Moderate, 2=High, 3=Highest)
4. Converts qualitative measurements to structured numeric features
5. Handles multiple reference ranges for calcitonin levels across studies
6. Engineers derived features (age groups, nodule presence, variant-specific interactions)
7. Generates two datasets:
   - `data/ret_multivariant_training_data.csv`: Original 78 patients from literature
   - `data/ret_multivariant_expanded_training_data.csv`: Expanded with synthetic controls
   - `data/ret_multivariant_case_control_dataset.csv`: Further expanded with variant-matched controls

### Important Notes on Data Quality

- **Multi-Variant Dataset:** Includes 11 different RET variants with varying penetrance and risk profiles
- **Risk Stratification:** Variants classified by ATA guidelines (Level 1/2/3)
- **Incomplete Penetrance:** Not all carriers develop MTC; penetrance varies by variant
- **Variable Follow-up:** Some carriers elected surveillance over prophylactic surgery
- **Age-Dependent Risk:** Penetrance increases with age, reflected in age-stratified features
- **Variant-Specific Patterns:** High-risk variants (C634\*) show different clinical patterns than moderate-risk (K666N, L790F)
- **Study Heterogeneity:** Different studies used different calcitonin reference ranges and screening protocols

**Key features:**

- **End-to-end pipeline** managed by `main.py`, coordinating all major steps automatically.
- **Multiple ML algorithms:** Support for Logistic Regression, Random Forest, XGBoost, and LightGBM models.
- **Model comparison mode:** Run all models simultaneously and compare performance metrics in a formatted table.
- **Dataset comparison mode:** Compare model performance on expanded dataset (with SMOTE and control cases) vs original paper data.
- **Automated data creation and expansion:** Scripts extract and structure relevant research data, and generate synthetic control samples to augment the dataset for robust modeling.
- **Comprehensive statistical analysis:** Automatic generation of descriptive statistics and visualization of the dataset for informed modeling.
- **Advanced model development:** Cross-validation and adaptive SMOTE balancing to handle class imbalance across all model types.
- **Clinical risk stratification:** 4-tier risk assessment (Low/Moderate/High/Very High) for actionable clinical decision-making.
- **Artifacts generated:** Processed datasets and trained model files, usable for risk scoring new patients with relevant clinical/genetic data.

**Pipeline steps (as run by `main.py`):**

1. **create_datasets.py:** Loads patient data from JSON files in [dataset/](dataset/) folder and formats into CSVs (78 patients from 4 studies, 11 variants).
2. **data_analysis.py:** Computes descriptive statistics, generates variant-specific visualizations and risk-stratified analyses.
3. **data_expansion.py:** Produces variant-matched synthetic control samples to improve model balance.
4. **train_model.py:** Trains models with variant features, cross-validation, SMOTE balancing, and threshold optimization.
5. **test_model.py:** Evaluates the model on test data with variant-specific risk stratification and comprehensive metrics.
6. **Artifact summary:** Includes `ret_multivariant_training_data.csv`, `ret_multivariant_expanded_training_data.csv`, `ret_multivariant_case_control_dataset.csv`, `model.pkl`.

**Advanced features:**

- **Data Leakage Prevention:** SMOTE applied after train/test split to ensure realistic evaluation
- **Feature Engineering:** Polynomial features (age²) and interactions (calcitonin×age, risk×age, nodule_severity)
- **Variant-Aware Modeling:** One-hot encoding of 11 RET variants + risk level stratification
- **Constant Feature Removal:** Automatic detection and removal of non-informative features
- **Risk Stratification:** 4-tier system for clinical decision support instead of binary classification
- **Comprehensive Metrics:** ROC-AUC, F1-Score, Average Precision Score, and confidence intervals

**Typical features used:**

- Age at diagnosis/intervention and derived features
- **RET variant type** (K666N, C634R, C634Y, L790F, etc.) - one-hot encoded
- **ATA risk level** (1=Moderate, 2=High, 3=Highest) - ordinal feature
- Calcitonin levels and elevation status
- Thyroid nodule characteristics
- Family history of MTC
- Clinical markers (pheochromocytoma, hyperparathyroidism)
- **Variant-specific interactions** (risk×calcitonin, risk×age)

**Clinical Use Case:**

- **Screening Tool:** Optimized for high sensitivity (catches all MTC cases)
- **Risk Stratification:** Provides actionable monitoring recommendations
- **Research Tool:** Validated on small datasets typical of rare genetic conditions

</details>

## Getting Started

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/ArjunCodess/men2-predictor.git
   cd men2-predictor
   ```

2. Create and activate virtual environment:

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: ./venv/Scripts/activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Project Structure

```
men2-predictor/
├── data/                                             # Processed datasets
│   ├── ret_multivariant_training_data.csv            # Original 78 patients
│   ├── ret_multivariant_expanded_training_data.csv   # Expanded with synthetic controls
│   └── ret_multivariant_case_control_dataset.csv     # Further expanded dataset
├── dataset/                                          # Raw study data (JSON)
│   ├── study_1.json
│   ├── study_2.json
│   ├── study_3.json
│   └── study_4.json
├── models/                                           # ML model implementations
│   ├── base_model.py
│   ├── random_forest_model.py
│   ├── lightgbm_model.py
│   ├── xgboost_model.py
│   └── logistic_regression_model.py
├── results/                                          # Test results and logs
├── src/                                              # Pipeline scripts
│   ├── create_datasets.py
│   ├── data_analysis.py
│   ├── data_expansion.py
│   ├── train_model.py
│   └── test_model.py
├── main.py                                           # Main pipeline entry point
└── requirements.txt
```

## Usage

### Basic Usage

Run the complete pipeline:

```sh
python main.py
```

This executes all stages: data preparation, analysis, expansion, training, and testing.

### Model Selection (`--m`)

Choose which model to train:

- `l` or `logistic`: Logistic Regression (default)
- `r` or `random_forest`: Random Forest ⭐ **Recommended**
- `x` or `xgboost`: XGBoost
- `g` or `lightgbm`: LightGBM ⭐ **Recommended**
- `a` or `all`: Run all models and compare

### Dataset Selection (`--d`)

Choose which dataset to use:

- `o` or `original`: Original 78 patients (no synthetic data) ⭐ **Recommended for clinical use**
- `e` or `expanded`: Expanded with synthetic controls + SMOTE (default)
- `b` or `both`: Run on both datasets for comparison

### Examples

```sh
# ⭐ RECOMMENDED FOR CLINICAL USE: Random Forest on original data
python main.py --m=random_forest --d=original

# ⭐ ALTERNATIVE RECOMMENDED: LightGBM on original data  
python main.py --m=lightgbm --d=original

# Compare all models on original dataset (identify best performer)
python main.py --m=all --d=original

# Demonstrate recall drop from synthetic augmentation (research use)
python main.py --m=random_forest --d=both

# ⚠️ NOT RECOMMENDED: Expanded dataset (lower recall)
python main.py --m=random_forest --d=expanded  # Only for research comparison
```

### Model Comparison Mode

When using `--m=all`, the pipeline:

1. Runs data preparation once (shared across models)
2. Trains and tests all four model types sequentially
3. Saves detailed logs to `results/logs/`
4. Displays comprehensive comparison table

### Dataset Comparison Mode

When using `--d=both`, the pipeline:

1. Runs the model on expanded dataset (synthetic + SMOTE)
2. Runs the same model on original dataset
3. Generates separate results files
4. Displays comparison table showing performance differences

This mode clearly demonstrates the recall degradation from synthetic augmentation.

### Output Files

**Model Files:**

- `saved_models/{model_type}_{dataset_type}_model.pkl`

**Results:**

- `results/{model_type}_{dataset_type}_test_results.txt`

**Logs (when using --m=all or --d=both):**

- `results/logs/{model_type}_{dataset_type}_training.log`
- `results/logs/{model_type}_{dataset_type}_testing.log`

## Technical Details

<details>
<summary><b>Feature Engineering</b></summary>

**Demographic Features:**

- Age at clinical evaluation (years)
- Gender (binary)
- Age groups (young/middle/elderly/very_elderly)

**Genetic Features:**

- RET variant (one-hot encoded across 11 variants)
- ATA risk level (ordinal: 1=Moderate, 2=High, 3=Highest)

**Biomarker Features:**

- Calcitonin elevation status (binary)
- Calcitonin level (numeric, pg/mL)

**Clinical Features:**

- Thyroid nodules presence
- Multiple nodules indicator
- Family history of MTC
- Pheochromocytoma presence
- Hyperparathyroidism presence

**Derived Features:**

- Polynomial features (age²)
- Interactions (calcitonin×age, risk×age, nodule_severity)
- Variant-specific risk interactions

</details>

<details>
<summary><b>Pipeline Steps</b></summary>

1. **create_datasets.py:** Loads patient data from JSON files, formats into CSV (78 patients, 11 variants)
2. **data_analysis.py:** Computes descriptive statistics, generates visualizations
3. **data_expansion.py:** Produces variant-matched synthetic control samples (optional)
4. **train_model.py:** Trains models with cross-validation, SMOTE balancing, threshold optimization
5. **test_model.py:** Evaluates models with comprehensive metrics and risk stratification

</details>

<details>
<summary><b>Model Architecture</b></summary>

**Supported Models:**

- Logistic Regression (baseline)
- Random Forest (ensemble, recommended)
- XGBoost (gradient boosting)
- LightGBM (gradient boosting, recommended)

**Training Configuration:**

- Cross-validation for hyperparameter tuning
- SMOTE balancing (applied after train/test split to prevent data leakage)
- Threshold optimization for clinical use cases
- Variant-aware feature encoding

**Evaluation Metrics:**

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Average Precision Score
- Confidence intervals
- Risk-stratified performance

</details>

<details>
<summary><b>Dataset Characteristics</b></summary>

**Original Dataset:**

- 78 confirmed RET germline mutation carriers
- 11 RET variants (K666N, L790F, Y791F, V804M, S891A, C634R, C634Y, C634W, C618S, C630R, C620Y)
- Age range: 5-90 years
- Mixed gender distribution
- ATA risk levels: Level 1 (Moderate), Level 2 (High), Level 3 (Highest)

**Expanded Dataset:**

- Original 78 patients + synthetic variant-matched controls
- Literature-based synthetic cases for improved balance
- SMOTE applied during training

**Data Quality Notes:**

- Multi-variant dataset includes varying penetrance and risk profiles
- Incomplete penetrance: not all carriers develop MTC
- Age-dependent risk: penetrance increases with age
- Variant-specific patterns: high-risk variants (C634\*) show different clinical patterns
- Study heterogeneity: different calcitonin reference ranges across studies

</details>

## Limitations

This study has several limitations that should be considered:

1. **Small sample size**: 78 patients is typical for rare genetic conditions but limits statistical power
2. **Retrospective data**: Extracted from published case series, not prospective validation
3. **Study heterogeneity**: Different calcitonin reference ranges and protocols across 4 studies
4. **Limited diversity**: Primarily European descent patients; generalizability to other populations unknown
5. **No external validation**: Performance validated on held-out data from same studies, not independent cohorts

**However**: These limitations are representative of rare disease ML challenges. Our finding (synthetic data harm) is strengthened by the fact that it persists across models and datasets.

**Next steps**: Prospective validation in clinical setting with multi-center collaboration.

## License

This project is licensed under the MIT License.

## Authors

**Arjun Vijay Prakash**

## Acknowledgements

Thanks to open source communities and packages including scikit-learn, pandas, numpy, matplotlib, seaborn, joblib, and imbalanced-learn for making data science and reproducibility accessible.

Special thanks to the authors of the research studies that provided clinical data:

- JCEM Case Reports (2025) - RET K666N carriers
- EDM Case Reports (2024) - RET K666N carriers
- Xu et al. Thyroid (2016) - RET K666N carriers
- European Journal of Endocrinology (2006) - Multi-variant RET carriers
