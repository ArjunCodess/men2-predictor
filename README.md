# MEN2 Predictor ML Model

A machine learning pipeline for predicting Multiple Endocrine Neoplasia type 2 (MEN2) syndrome, with a focus on the RET K666N gene mutation, using clinical and genetic features.

This project automates data preparation, exploratory analysis, dataset expansion, model training, evaluation, and artifact generation to streamline MEN2 risk prediction based on available research and synthetic datasets.

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

This project uses clinical data extracted from three peer-reviewed research studies on RET K666N germline mutation carriers:

1. **Study 1 - JCEM Case Reports (March 2025)**
   - Title: "Medullary Thyroid Carcinoma and Clinical Outcomes in Heterozygous Carriers of the RET K666N Germline Pathogenic Variant"
   - 4 patients (family cluster with index case)

2. **Study 2 - EDM Case Reports (September 2024)**
   - Title: "MEN2 phenotype in a family with germline heterozygous rare RET K666N variant"
   - DOI: 10.1530/EDM-24-0009
   - 4 patients (family cluster with MEN2 features)

3. **Study 3 - Thyroid Journal (2016)**
   - Title: "Medullary Thyroid Carcinoma Associated with Germline RETK666N Mutation"
   - DOI: 10.1089/thy.2016.0374
   - 24 patients across 8 families (including probands and cascade-tested relatives)

### Dataset Characteristics

**Original Paper Dataset:** 32 confirmed RET K666N carriers
- Age range: 5-90 years
- Gender distribution: Mixed (Male/Female)
- All patients carry the heterozygous RET K666N (c.1998G>T, p.Lys666Asn) pathogenic variant
- MTC diagnosis cases: 11/32 (34.4%)
- C-cell disease (MTC + C-cell hyperplasia): 13/32 (40.6%)

**Expanded Dataset:** Original 32 patients + synthetic controls
- Includes literature-based synthetic cases for improved model balance
- Synthetic controls generated using clinical distributions from research data
- Enhanced with SMOTE (Synthetic Minority Over-sampling Technique) during training

### Clinical Features

The dataset includes the following structured clinical and genetic features:

**Demographic Features:**
- `age`: Age at clinical evaluation (years)
- `gender`: Biological sex (0=Female, 1=Male)
- `age_group`: Categorized age ranges (young/middle/elderly/very_elderly)

**Genetic Features:**
- `ret_k666n_positive`: RET K666N mutation status (all patients = 1)

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

### Data Processing Pipeline

The [create_datasets.py](src/create_datasets.py) script:
1. Extracts patient data from nested research paper structures
2. Converts qualitative measurements to structured numeric features
3. Handles multiple reference ranges for calcitonin levels across studies
4. Engineers derived features (age groups, nodule presence)
5. Generates two datasets:
   - `data/ret_k666n_training_data.csv`: Original 32 patients from literature
   - `data/men2_case_control_dataset.csv`: Expanded with synthetic controls

### Important Notes on Data Quality

- **Small Sample Size:** Reflects the rarity of K666N variant (one of the least common RET mutations)
- **Incomplete Penetrance:** Not all K666N carriers develop MTC, consistent with literature
- **Variable Follow-up:** Some carriers elected surveillance over prophylactic surgery
- **Age-Dependent Risk:** Penetrance increases with age, reflected in age-stratified features
- **No PHEO/PHPT:** Unlike high-risk RET mutations, K666N rarely presents with pheochromocytoma or hyperparathyroidism

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
1. **create_datasets.py:** Extracts and formats case/control and RET K666N mutation data into CSVs (32 patients from 3 studies).
2. **data_analysis.py:** Computes descriptive statistics and generates visualizations to aid in understanding cohort differences.
3. **data_expansion.py:** Produces synthetic control samples to improve model balance.
4. **train_model.py:** Trains a logistic regression model with cross-validation, SMOTE balancing, and threshold optimization.
5. **test_model.py:** Evaluates the model on test data with risk stratification and comprehensive metrics.
6. **Artifact summary:** Includes `ret_k666n_training_data.csv`, `ret_k666n_expanded_training_data.csv`, `men2_case_control_dataset.csv`, `model.pkl`.

**Advanced features:**
- **Data Leakage Prevention:** SMOTE applied after train/test split to ensure realistic evaluation
- **Feature Engineering:** Polynomial features (age²) and interactions (calcitonin×age, nodule_severity)
- **Constant Feature Removal:** Automatic detection and removal of non-informative features
- **Risk Stratification:** 4-tier system for clinical decision support instead of binary classification
- **Comprehensive Metrics:** ROC-AUC, F1-Score, Average Precision Score, and confidence intervals

**Typical features used:**
- Age at diagnosis/intervention and derived features
- Calcitonin levels and elevation status  
- Thyroid nodule characteristics
- Family history of MTC
- RET mutation status (with special focus on K666N variant)
- Clinical markers (pheochromocytoma, hyperparathyroidism)

**Clinical Use Case:**
- **Screening Tool:** Optimized for high sensitivity (catches all MTC cases)
- **Risk Stratification:** Provides actionable monitoring recommendations
- **Research Tool:** Validated on small datasets typical of rare genetic conditions

## Getting Started

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
- Original paper dataset: `data/ret_k666n_training_data.csv` (32 patients)
- Expanded dataset: `data/men2_case_control_dataset.csv` (with synthetic controls)

## License

This project is licensed under the MIT License.

## Authors

**Arjun Vijay Prakash**

## Acknowledgements

Thanks to open source communities and packages including scikit-learn, pandas, numpy, matplotlib, seaborn, joblib, and imbalanced-learn for making data science and reproducibility accessible.

Additional credit to researchers whose data informed the synthetic controls and simulations in this tool. Special thanks to the authors of the JCEM Case Reports (2025), EDM Case Reports (2024), and Xu et al. Thyroid (2016) studies for providing clinical data on RET K666N carriers.