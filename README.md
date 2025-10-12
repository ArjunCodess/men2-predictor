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

This repository provides a reproducible machine learning pipeline to predict MEN2 risk, primarily leveraging Python’s scientific stack.

**Key features:**
- **End-to-end pipeline** managed by `main.py`, coordinating all major steps automatically.
- **Automated data creation and expansion:** Scripts extract and structure relevant research data, and generate synthetic control samples to augment the dataset for robust modeling.
- **Comprehensive statistical analysis:** Automatic generation of descriptive statistics and visualization of the dataset for informed modeling.
- **Model development:** Logistic regression with cross-validation to distinguish MEN2 cases from controls.
- **Artifacts generated:** Processed datasets and a trained `model.pkl`, usable for risk scoring new patients with relevant clinical/genetic data.

**Pipeline steps (as run by `main.py`):**
1. **create_datasets.py:** Extracts and formats case/control and RET K666N mutation data into CSVs.
2. **data_analysis.py:** Computes descriptive statistics and generates visualizations to aid in understanding cohort differences.
3. **data_expansion.py:** Produces synthetic control samples to improve model balance.
4. **train_model.py:** Trains a logistic regression model with cross-validation and saves it.
5. **test_model.py:** Evaluates the model on test data and provides performance metrics.
6. **Artifact summary:** Includes `ret_k666n_training_data.csv`, `ret_k666n_expanded_training_data.csv`, `men2_case_control_dataset.csv`, `model.pkl`.

**Typical features used:**
- Age at diagnosis/intervention
- Calcitonin levels
- RET mutation status (with special focus on K666N variant)
- Clinical markers relevant to MEN2 syndrome

## Getting Started

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/your-username/men2-predictor.git
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

## License

This project is licensed under the MIT License.

## Authors

**Arjun Vijay Prakash**

## Acknowledgements

Thanks to open source communities and packages including scikit-learn, pandas, numpy, matplotlib, seaborn, and joblib for making data science and reproducibility accessible.
Additional credit to researchers whose data informed the synthetic controls and simulations in this tool.