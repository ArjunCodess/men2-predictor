# MEN2 Predictor ML Model

A machine learning pipeline for predicting Multiple Endocrine Neoplasia type 2 (MEN2) syndrome across multiple RET gene variants, using clinical and genetic features.

This project automates data preparation, exploratory analysis, dataset expansion, model training, evaluation, and artifact generation to streamline MEN2 risk prediction based on available research and synthetic datasets. The model supports **11 different RET variants** including K666N, C634R, C634Y, C634W, and others, with ATA risk level stratification.

## Table of Contents

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)  
- [Usage](#usage)
- [License](#license)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

## About The Project

MEN2 is a rare hereditary cancer syndrome associated with RET gene mutations.

This repository provides a reproducible machine learning pipeline to predict MEN2 risk, primarily leveraging Python's scientific stack.

## Data Sources and Structure

### Research Data Sources

This project uses clinical data extracted from four peer-reviewed research studies covering multiple RET germline mutation carriers:

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

**Multi-Variant Dataset:** 74 confirmed RET germline mutation carriers across 11 variants
- **Studies 1-3 (K666N cohort):** 28 patients (after deduplication)
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

**Expanded Dataset:** Original 74 patients + synthetic variant-matched controls
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
2. Extracts and combines data from multiple research studies (74 patients, 11 variants)
3. Maps each variant to ATA risk level (1=Moderate, 2=High, 3=Highest)
4. Converts qualitative measurements to structured numeric features
5. Handles multiple reference ranges for calcitonin levels across studies
6. Engineers derived features (age groups, nodule presence, variant-specific interactions)
7. Generates two datasets:
   - `data/ret_multivariant_training_data.csv`: Original 74 patients from literature
   - `data/ret_multivariant_expanded_training_data.csv`: Expanded with synthetic controls
   - `data/ret_multivariant_case_control_dataset.csv`: Further expanded with variant-matched controls

### Important Notes on Data Quality

- **Multi-Variant Dataset:** Includes 11 different RET variants with varying penetrance and risk profiles
- **Risk Stratification:** Variants classified by ATA guidelines (Level 1/2/3)
- **Incomplete Penetrance:** Not all carriers develop MTC; penetrance varies by variant
- **Variable Follow-up:** Some carriers elected surveillance over prophylactic surgery
- **Age-Dependent Risk:** Penetrance increases with age, reflected in age-stratified features
- **Variant-Specific Patterns:** High-risk variants (C634*) show different clinical patterns than moderate-risk (K666N, L790F)
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
1. **create_datasets.py:** Loads patient data from JSON files in [dataset/](dataset/) folder and formats into CSVs (74 patients from 4 studies, 11 variants).
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

## Getting Started

### Project Structure

```
men2-predictor/
├── charts
│   ├── age_histograms.png
│   ├── calcitonin_boxplots.png
│   ├── calcitonin_by_variant.png
│   ├── feature_distributions.png
│   ├── risk_level_analysis.png
│   └── variant_distribution.png
├── data
│   ├── ret_multivariant_case_control_dataset.csv
│   ├── ret_multivariant_expanded_training_data.csv
│   └── ret_multivariant_training_data.csv
├── dataset
│   ├── literature_data.json
│   ├── mutation_characteristics.json
│   ├── study_1.json
│   ├── study_2.json
│   ├── study_3.json
│   └── study_4.json
├── models
│   ├── __init__.py
│   ├── base_model.py
│   ├── lightgbm_model.py
│   ├── logistic_regression_model.py
│   ├── random_forest_model.py
│   └── xgboost_model.py
├── results
│   ├── lightgbm_expanded_test_results.txt
│   ├── lightgbm_original_test_results.txt
│   ├── lightgbm_test_results.txt
│   ├── logistic_expanded_test_results.txt
│   ├── logistic_original_test_results.txt
│   ├── logistic_test_results.txt
│   ├── random_forest_expanded_test_results.txt
│   ├── random_forest_original_test_results.txt
│   ├── random_forest_test_results.txt
│   ├── xgboost_expanded_test_results.txt
│   ├── xgboost_original_test_results.txt
│   └── xgboost_test_results.txt
├── src
│   ├── create_datasets.py
│   ├── data_analysis.py
│   ├── data_expansion.py
│   ├── test_model.py
│   └── train_model.py
├── .gitignore
├── LICENSE
├── main.py
├── README.md
└── requirements.txt
```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/ArjunCodess/men2-predictor.git
   ```
2. Go to the project folder
   ```sh
   cd men2-predictor
   ```
3. Set up a virtual environment
   ```sh
   python -m venv venv
   ```
4. Activate the environment
   ```sh
   source venv/bin/activate  # On Windows use ./venv/Scripts/activate
   ```
5. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```

## Usage

Run the pipeline from the root of the project:

```sh
python main.py
```

- This will execute all pipeline stages in order.
- Intermediate and output files (such as cleaned datasets and `model.pkl`) will be generated in the project directory.
- Check the terminal/log output for detailed progress, statistics, and any encountered errors.

For detailed usage or to run a pipeline step separately, see each script (`create_datasets.py`, `train_model.py`, etc.).

### Model selection (--m)

You can choose which model to train and test using the `--m` argument:

### Dataset selection (--d)

You can choose which dataset to use for training and testing using the `--d` argument:

- `e` or `expanded`: expanded dataset with synthetic controls + SMOTE balancing (default)
- `o` or `original`: original paper dataset only (32 patients, no synthetic controls)
- `b` or `both`: run on both datasets for comparison

- `l` or `logistic`: logistic regression (default)
- `r` or `random_forest`: random forest
- `x` or `xgboost`: xgboost
- `g` or `lightgbm`: lightgbm
- `a` or `all`: **run all models and compare results**

Examples:

```sh
# run full pipeline with logistic regression (default)
python main.py
python main.py --m=l
python main.py --m=logistic

# run full pipeline with random forest
python main.py --m=r
python main.py --m=random_forest

# run full pipeline with xgboost
python main.py --m=x
python main.py --m=xgboost

# run full pipeline with lightgbm
python main.py --m=g
python main.py --m=lightgbm

# run ALL models and compare performance in a table
python main.py --m=all
python main.py --m=a

# Dataset selection examples
# run logistic regression on expanded dataset (default)
python main.py --m=l --d=e
python main.py  # same as above (both are defaults)

# run random forest on original paper data only
python main.py --m=r --d=o

# run xgboost on BOTH datasets for comparison
python main.py --m=x --d=both

# run ALL models on BOTH datasets (comprehensive comparison - 8 total runs)
python main.py --m=all --d=both

# train only
python src/train_model.py --m=l --d=e
python src/train_model.py --m=r --d=o
python src/train_model.py --m=x
python src/train_model.py --m=g

# test only (expects corresponding saved model in project root)
python src/test_model.py --m=l --d=e
python src/test_model.py --m=r --d=o
python src/test_model.py --m=x
python src/test_model.py --m=g
```

### Model Comparison Mode

When using `--m=all`, the pipeline will:
1. Run data preparation steps once (shared across all models)
2. Train and test all four model types sequentially (one at a time)
3. Save detailed execution logs to `results/logs/` for each step
4. Save individual test results to `results/{model_type}_{dataset_type}_test_results.txt`
5. Display a comprehensive comparison table with all metrics

**Log Files Generated:**
- Data preparation: `results/logs/data_preparation_step{1,2,3}.log`
- Model training: `results/logs/{model_type}_{dataset_type}_training.log`
- Model testing: `results/logs/{model_type}_{dataset_type}_testing.log`

This approach ensures:
- All detailed output is preserved in log files for later review
- Console output remains clean and focused on progress and metrics
- You can debug issues by checking the specific log file for each step
- Each model's training and testing output is kept separate for easy comparison

This mode is ideal for:
- **Model selection:** Identify which algorithm performs best on your data
- **Dataset comparison:** Compare performance on expanded vs original datasets
- **Performance benchmarking:** Compare metrics across different approaches
- **Research and reporting:** Generate comprehensive comparison data

### Dataset Comparison Mode

When using `--d=both`, the pipeline will:
1. Run the selected model(s) on the **expanded dataset** (with synthetic controls + SMOTE)
2. Run the same model(s) on the **original dataset** (paper data only)
3. Generate separate model files and results for each dataset
4. Display a comparison table showing performance differences

This helps you understand:
- How synthetic controls and SMOTE affect model performance
- Whether your model overfits to synthetic data
- Which dataset configuration works best for your use case

### Artifacts

**Model Files:**
- Logistic regression model saved to `saved_models/logistic_regression_{dataset_type}_model.pkl`
- Random forest model saved to `saved_models/random_forest_{dataset_type}_model.pkl`
- XGBoost model saved to `saved_models/xgboost_{dataset_type}_model.pkl`
- LightGBM model saved to `saved_models/lightgbm_{dataset_type}_model.pkl`

Where `{dataset_type}` is either `expanded` or `original`

**Results & Metrics:**
- Test results saved to `results/{model_type}_{dataset_type}_test_results.txt`

**Execution Logs (when using --m=all or --d=both):**
- Data preparation logs in `results/logs/data_preparation_step*.log`
- Training logs in `results/logs/{model_type}_{dataset_type}_training.log`
- Testing logs in `results/logs/{model_type}_{dataset_type}_testing.log`

**Data Files:**
- Original multi-variant dataset: `data/ret_multivariant_training_data.csv` (74 patients, 11 variants)
- Expanded dataset: `data/ret_multivariant_expanded_training_data.csv` (with synthetic variant-matched controls)
- Case-control dataset: `data/ret_multivariant_case_control_dataset.csv` (further expanded with SMOTE-ready controls)

## License

This project is licensed under the MIT License.

## Authors

**Arjun Vijay Prakash**

## Acknowledgements

Thanks to open source communities and packages including scikit-learn, pandas, numpy, matplotlib, seaborn, joblib, and imbalanced-learn for making data science and reproducibility accessible.

Additional credit to researchers whose data informed the synthetic controls and simulations in this tool. Special thanks to the authors of the JCEM Case Reports (2025), EDM Case Reports (2024), and Xu et al. Thyroid (2016) studies for providing clinical data on RET K666N carriers.