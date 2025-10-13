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

**Key features:**
- **End-to-end pipeline** managed by `main.py`, coordinating all major steps automatically.
- **Automated data creation and expansion:** Scripts extract and structure relevant research data, and generate synthetic control samples to augment the dataset for robust modeling.
- **Comprehensive statistical analysis:** Automatic generation of descriptive statistics and visualization of the dataset for informed modeling.
- **Advanced model development:** Logistic regression with cross-validation and SMOTE balancing to handle class imbalance.
- **Clinical risk stratification:** 4-tier risk assessment (Low/Moderate/High/Very High) for actionable clinical decision-making.
- **Artifacts generated:** Processed datasets and a trained `model.pkl`, usable for risk scoring new patients with relevant clinical/genetic data.

**Pipeline steps (as run by `main.py`):**
1. **create_datasets.py:** Extracts and formats case/control and RET K666N mutation data into CSVs (8 patients from 2 studies).
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

- `l` or `logistic`: logistic regression (default)
- `r` or `random_forest`: random forest
- `x` or `xgboost`: xgboost
- `g` or `lightgbm`: lightgbm

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

# train only
python src/train_model.py --m=l
python src/train_model.py --m=r
python src/train_model.py --m=x
python src/train_model.py --m=g

# test only (expects corresponding saved model in project root)
python src/test_model.py --m=l
python src/test_model.py --m=r
python src/test_model.py --m=x
python src/test_model.py --m=g
```

Artifacts:

- Logistic regression model saved to `logistic_regression_model.pkl`
- Random forest model saved to `random_forest_model.pkl`
- XGBoost model saved to `xgboost_model.pkl`
- LightGBM model saved to `lightgbm_model.pkl`

## License

This project is licensed under the MIT License.

## Authors

**Arjun Vijay Prakash**

## Acknowledgements

Thanks to open source communities and packages including scikit-learn, pandas, numpy, matplotlib, seaborn, joblib, and imbalanced-learn for making data science and reproducibility accessible.

Additional credit to researchers whose data informed the synthetic controls and simulations in this tool. Special thanks to the authors of the JCEM Case Reports (2025) and EDM Case Reports (2024) studies for providing clinical data on RET K666N carriers.