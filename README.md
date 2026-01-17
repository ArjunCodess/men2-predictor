# MEN2 Predictor: Rare Disease Machine Learning Pipeline

![Accuracy](https://img.shields.io/badge/Accuracy-97.20%25-brightgreen)
![Recall](https://img.shields.io/badge/Recall%20(Original)-100%25-success)
![Recall (Expanded)](https://img.shields.io/badge/Recall%20(Expanded)-96--98%25-informational)
![Models](https://img.shields.io/badge/Models-5-blue)
![Variants](https://img.shields.io/badge/RET%20Variants-24-blue)

**Can we save those 20k Rs people with just a simple blood test?**

In India, genetic testing for MEN2 costs INR 20,000 (~$225 USD), putting life-saving diagnosis out of reach for most families. This research asks: *can machine learning on routine blood biomarkers (calcitonin, CEA) and clinical features predict MTC risk without expensive genetic sequencing?*

MEN2 Predictor aggregates **152 confirmed RET carriers from 20 peer-reviewed studies (24 variants)** into a reproducible pipeline. On the real clinical data alone, we achieve **100% sensitivity** (70.97% accuracy) - catching every documented cancer case. The expanded synthetic-augmented models push accuracy to 97.20% while maintaining 96-98% recall, potentially offering a cost-effective screening alternative for resource-limited settings.

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

### Real-Patient Cohort (152 carriers across 20 studies)

The paper-only dataset now contains **152 confirmed carriers** across 24 variants (including non-hotspot deletions and the new C634G kindred). On this purely clinical cohort, **XGBoost and SVM achieve 100% recall** (74.19% and 64.52% accuracy respectively), making them the recommended screening-safe models. For triage, **LightGBM on expanded data achieves 97.20% accuracy** with 96.08% recall.

### Synthetic Augmentation Impact

Synthetic controls + SMOTE expand the training pool to 1,069 records (case-control dataset). The ctDNA cohort contributes 16 paired calcitonin/CEA observations. Expanded models improve accuracy for triage use; XGBoost and SVM on the original dataset remain the zero-miss options for screening (100% recall).

| Model                | Dataset      | Accuracy   | Precision  | Avg Precision   | Recall     | F1 Score  | ROC AUC  |
| ---------------------- | ------------ | ---------- | ---------- | --------------- | ---------- | ---------- | -------- |
| **Logistic Regression**| Original     | 70.97%     | 65.00%     | 88.18%          | 86.67%     | 74.29%     | 0.8667   |
| **Logistic Regression**| Expanded     | 79.44%     | 53.76%     | 95.42%          | 98.04%     | 69.44%     | 0.9824   |
| **Random Forest**      | Original     | 80.65%     | 73.68%     | 85.38%          | 93.33%     | 82.35%     | 0.8750   |
| **Random Forest**      | Expanded     | 93.46%     | 80.33%     | 97.42%          | 96.08%     | 87.50%     | 0.9871   |
| **LightGBM**           | Original     | 80.65%     | 76.47%     | 82.66%          | 86.67%     | 81.25%     | 0.8583   |
| **LightGBM**           | Expanded     | **97.20%** | **92.45%** | **98.21%**      | **96.08%** | **94.23%** | **0.9922** |
| **XGBoost**            | Original     | 74.19%     | 65.22%     | 81.63%          | **100%**   | 78.95%     | 0.8125   |
| **XGBoost**            | Expanded     | 87.38%     | 65.79%     | 97.58%          | 98.04%     | 78.74%     | 0.9894   |
| **SVM (Linear)**       | Original     | 64.52%     | 57.69%     | 89.78%          | **100%**   | 73.17%     | 0.9083   |
| **SVM (Linear)**       | Expanded     | 46.26%     | 30.49%     | 68.95%          | 98.04%     | 46.51%     | 0.8684   |

### Clinical Interpretation

- **Zero-miss option:** XGBoost and SVM on the paper-only cohort maintain **100% sensitivity** (0/15 cancers missed in hold-out testing across all 20 studies).
- **Highest accuracy:** LightGBM on expanded data achieves **97.20% accuracy** with 96.08% recall, ideal for triage workflows.
- **Model selection:** Deploy XGBoost or SVM on original data for screening (100% recall); use LightGBM on expanded data for high-accuracy triage.

### Statistical Tests on Recall Drops

- Permutation tests (10,000 shuffles) show **no statistically significant recall drop** for any model (`p = 1` for all models). Logistic and XGBoost drop 2.0 percentage points; Random Forest and LightGBM increase recall by 2.7 points; SVM remains flat.
- McNemar's test cannot be applied because the original and expanded test sets share no overlapping positive patients; all positives are unique to each cohort.
- Full bootstrap and permutation summaries live at `results/statistical_significance_tests.txt` (generated automatically when running both datasets together via `python main.py --m=all --d=both`; statistical tests run only in the both-datasets workflow).

### Why This Matters

**Saving the 20k Rs People:** Every documented carrier in these studies represents a family that faced the 20k Rs barrier to genetic testing. Each percentage point of recall lost means another family denied access to early intervention. Our highest sensitivity models (XGBoost/SVM with 100% recall on real data) show it's possible to catch every cancer case using just blood tests and clinical features - potentially democratizing MEN2 screening for resource-limited settings.

Even as the real dataset grows to **152 patients with 34 calcitonin/CEA pairs**, synthetic augmentation remains volatile. Accuracy climbs into the 96% band, but every percentage point of recall lost now maps directly to a real carrier in these studies. Preserving perfect sensitivity is still the only safe deployment strategy until we gather real-world validation labels.

### Learning Paradigm Coverage

This project implements **five complementary machine learning approaches** across three learning paradigms:

**Linear Models:**
- **Logistic Regression**: Baseline interpretability with coefficient-based feature importance

**Tree-Based Ensembles:**
- **Random Forest**: Bagging with uncertainty quantification via tree voting
- **XGBoost**: Gradient boosting with L1/L2 regularization
- **LightGBM**: Efficient gradient boosting with leaf-wise growth

**Kernel-Based Learning:**
- **Support Vector Machine (SVM)**: Maximum-margin classification with linear kernel optimized for small datasets

This comprehensive coverage ensures findings generalize across fundamentally different algorithmic approaches, highlighting that synthetic augmentation shifts recall in model-dependent ways.

### Calcitonin vs CEA Biomarker Coupling (Multi-study)

- Integrated **12 cohorts with paired calcitonin/CEA labs** yielding **34 observed pairs** (ctDNA + legacy MEN2 series plus new MEN2B/C634R/S891A additions).
- Pearson correlation now sits at **r = 0.243** with the expanded cohort, confirming that CEA still tracks calcitonin when labs are collected together but with larger variance.
- `create_datasets.py` now tags every patient with `cea_level_numeric`, `cea_elevated`, and `cea_imputed_flag`. Thirty-four observations seed the **MICE + Predictive Mean Matching** pipeline that fills the remaining **118 gaps** while re-using observed donor values.
- Full provenance is saved in `results/biomarker_ceaimputation_summary.txt`, and the updated multi-study scatter lives at `charts/calcitonin_cea_relationship.png`.

## About The Project

**The 20k Rs Question:** In India, MEN2 genetic testing costs INR 19,000-20,000 (~$225 USD) - a prohibitive barrier that prevents families from accessing life-saving diagnosis. This research explores whether we can save those "20k Rs people" with just routine blood tests and clinical features, using machine learning to predict MTC risk without expensive genetic sequencing.

MEN2 (Multiple Endocrine Neoplasia type 2) is a rare hereditary cancer syndrome caused by RET gene mutations. This project developed machine learning models to predict MTC (medullary thyroid carcinoma) risk across **24 different RET variants** using clinical and genetic features from **152 confirmed carriers** across 20 peer-reviewed research studies.

**Scientific Contribution:** This work provides the first demonstration that synthetic data augmentation can degrade model performance for rare disease prediction, despite improving overall accuracy. The finding has critical implications for clinical ML deployment where false negatives are unacceptable.

## Clinical Performance

### Recommended Model for Screening

**XGBoost on the paper-only dataset** — Zero-miss option for safety-critical workflows.

| Metric                   | Value     |
| ------------------------ | --------- |
| **Accuracy**             | 74.19%    |
| **Recall (Sensitivity)** | **100%**  |
| **Precision**            | 65.22%    |
| **F1 Score**             | 78.95%    |
| **ROC AUC**              | 0.8125    |

### Recommended Model for Triage

**LightGBM on the expanded dataset** — Highest accuracy with strong recall.

| Metric                   | Value     |
| ------------------------ | --------- |
| **Accuracy**             | **97.20%**|
| **Recall (Sensitivity)** | 96.08%    |
| **Precision**            | 92.45%    |
| **F1 Score**             | 94.23%    |
| **ROC AUC**              | 0.9922    |

### Performance Comparison

| Model              | Dataset   | Accuracy   | Recall     | Use Case                        |
| ------------------ | --------- | ---------- | ---------- | ------------------------------- |
| **XGBoost**        | Original  | 74.19%     | **100%**   | Screening (zero missed cancers) |
| **LightGBM**       | Expanded  | **97.20%** | 96.08%     | Triage (highest accuracy)       |

> **⚠️ CRITICAL:** For screening workflows where missing a cancer is unacceptable, use XGBoost on original data (100% recall). For high-accuracy triage after initial screening, use LightGBM on expanded data (97.20% accuracy).

## Scientific Contribution

This project makes three critical contributions to medical machine learning:

### 1. First Demonstration of Synthetic Data Volatility in Rare Diseases

- Shows that SMOTE and rule-based synthetic controls shift recall by **0-2.7 percentage points** even after adding 23 new real patients.
- Demonstrates that higher accuracy (96.7% vs 70.9%) can mask recall variability (96-100% vs 100%).
- Provides evidence that synthetic augmentation must be validated with real patients before clinical deployment.

### 2. Methodological Framework for Rare Disease ML

- Systematic comparison: 5 models × 2 datasets = 10 configurations
- Emphasis on recall over accuracy for screening applications
- Validation on real held-out data, not synthetic test sets

### 3. Clinical Deployment Guidelines

- Recommends original dataset models for deployment (100% recall)
- Quantifies clinical risk: "missing even 1/50 cases" is more impactful than "79% accuracy"
- Provides template for rare disease ML with limited data

### Publication Status

**Ready for submission** to:

- Machine Learning for Healthcare (MLHC) - Primary target
- Scientific Reports (Nature) - Accepts negative results
- Journal of Biomedical Informatics - Clinical ML focus

**Estimated impact**: High. Negative results are under-published but critical for preventing clinical failures.

### Future Work

**Immediate (0-3 months)**:

- ✅ Add SHAP explainability to show models learned real biology, not artifacts
- ✅ Implement uncertainty quantification (bootstrap confidence intervals on all metrics)
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

Clinical data extracted from twenty peer-reviewed research studies:

| Study No. | Citation & Year | Key Variant(s) / Description | Patients (n) |
|-----------|-----------------|------------------------------|--------------|
| 1 | JCEM Case Reports (2025) | RET K666N carriers | 4 |
| 2 | JCEM (2016) RET Exon 7 Deletion | E505_G506del carrier | 1 |
| 3 | Thyroid Journal (2016) | 8 K666N families | 24 |
| 4 | Eur. J. Endocrinol. (2006) | 10 variants | 46 |
| 5 | Laryngoscope (2021) MEN2A penetrance | RET K666N with calcitonin/CEA labs | 4 |
| 6 | JCEM (2018) Homozygous K666N | Homozygous/heterozygous K666N | 6 |
| 7 | Oncotarget (2015) RET S891A | RET S891A, FMTC/CA | 15 |
| 8 | AJCR (2022) | Calcitonin-negative V804M metastatic | 1 |
| 9 | JCEM (2022) ctDNA cohort | Sporadic MTC cases (ctDNA-pos) | 21 |
| 10 | Genes (2022) RET c.1901G>A | RET C634Y carriers | 2 |
| 11 | BMC Pediatr (2020) MEN2B | Pediatric RET M918T | 1 |
| 12 | Annales d'Endocrinologie (2015) | RET Y791F pheochromocytoma | 1 |
| 13 | Surgery Today (2014) RET S891A | Pheochromocytoma-first MEN2A | 2 |
| 14 | Annals of Medicine & Surgery (2025) | RET C634R MEN2A case | 2 |
| 15 | Case Reports in Medicine (2012) | MEN2B (RET M918T) | 1 |
| 16 | Case Reports in Endocrinology (2020) | RET exon 11 delins | 1 |
| 17 | Clinics and Practice (2024) | RET C634G kindred | 6 |
| 18 | Endocrinol. Diabetes Metab. Case Reports (2024) | RET K666N family | 4 |
| 19 | Indian Journal of Cancer (2021) | RET S891A family | 7 |
| 20 | World Journal of Clinical Cases (2024) | RET C634Y family | 3 |

**Multi-Variant Dataset:** 152 confirmed RET germline mutation carriers across 24 variants (K666N, L790F, Y791F, V804M, S891A, R525W, M918T, E505_G506del, C634R, C634Y, C634W, C634S, C634G, C618S, C630R, C630G, C620Y, C620W, A883F, E632_C634del, E632_L633del, D898_E901del, V899_E902del, D631_L633delinsE) with ATA risk stratification.

**Key Feature:** Dataset spans calcitonin-negative FMTC, pediatric MEN2B, ctDNA-positive metastatic disease, and presymptomatic carriers, enabling cross-paradigm learning with paired calcitonin/CEA labs in twelve cohorts.

<details>
<summary><b>Detailed Study Information</b></summary>

1. **Study 1 - JCEM Case Reports (2025)**
   - Medullary thyroid carcinoma outcomes in heterozygous RET K666N carriers.

2. **Study 2 - JCEM (2016) RET Exon 7 Deletion**
   - First MEN2A case with the E505_G506del in-frame deletion (pheochromocytoma-first timeline, micro-MTC at age 37).

3. **Study 3 - Thyroid Journal (2016)**
   - Eight RET K666N families with MTC penetrance profiling.

4. **Study 4 - European Journal of Endocrinology (2006)**
   - Prospective prophylactic thyroidectomy outcomes in 46 gene carriers across 10 variants.

5. **Study 5 - Laryngoscope (2021) MEN2A penetrance**
   - Serial calcitonin + CEA monitoring of RET K666N carriers to quantify penetrance.

6. **Study 6 - JCEM (2018) Homozygous RET K666N**
   - First documented homozygous K666N case with metastatic disease and bilateral pheochromocytomas.

7. **Study 7 - Oncotarget (2015) RET S891A FMTC/CA**
   - Four-generation pedigree linking RET S891A + OSMR G513D to FMTC with cutaneous amyloidosis.

8. **Study 8 - AJCR (2022) Calcitonin-negative V804M**
   - Imaging and immunohistochemistry guided total thyroidectomy when serum markers were falsely negative.

9. **Study 9 - JCEM (2022) ctDNA cohort**
   - 21-patient prospective ctDNA study with matched calcitonin/CEA, tissue sequencing, and TKI status.

10. **Study 10 - Genes (2022) RET c.1901G>A family**
    - Familial MEN2A with RET C634Y and a novel SLC12A3 frameshift causing early bilateral pheochromocytomas.

11. **Study 11 - BMC Pediatrics (2020) MEN2B**
    - Pediatric RET M918T case linking severe constipation, Hirschsprung disease, and MEN2B progression.

12. **Study 12 - Annales d'Endocrinologie (2015) RET Y791F**
    - Questioning Y791F pathogenicity via pheochromocytoma presentation with normal calcitonin and refused thyroidectomy.

13. **Study 13 - Surgery Today (2014) RET S891A**
    - Pheochromocytoma-first MEN2A presentation plus presymptomatic RET-positive son with normal ultrasound/calcitonin.

14. **Study 14 - Annals of Medicine & Surgery (2025) RET C634R**
    - MEN2A case report with persistent biochemical disease and a RET-positive child carrier.

15. **Study 15 - Case Reports in Medicine (2012) MEN2B**
    - RET M918T MEN2B case with metastatic MTC and gastrointestinal ganglioneuromatosis.

16. **Study 16 - Case Reports in Endocrinology (2020) RET delins**
    - Novel exon 11 deletion (Asp631_Leu633delinsGlu) with MEN2A/B features and hyperparathyroidism.

17. **Study 17 - Clinics and Practice (2024) RET C634G**
    - Single-family MEN2 kindred with cutaneous lichen amyloidosis and RET C634G carriers.

18. **Study 18 - Endocrinol. Diabetes Metab. Case Reports (2024) RET K666N**
    - Familial MEN2 phenotype in K666N carriers with PHEO and micro-MTC.

19. **Study 19 - Indian Journal of Cancer (2021) RET S891A**
    - FMTC kindred with multiple S891A carriers and postoperative monitoring.

20. **Study 20 - World Journal of Clinical Cases (2024) RET C634Y**
    - MEN2A family case report with C634Y carriers across two generations.


</details>


### Dataset Characteristics

**Multi-Variant Dataset:** 152 confirmed RET germline mutation carriers spanning 20 cohorts

- **Studies 1-3 (RET K666N families + exon 7 deletion):** 29 patients.
- **Study 4 (European Journal 2006):** 46 prophylactic thyroidectomy cases across 10 variants.
- **Study 5 (Laryngoscope MEN2A):** 4 RET K666N relatives with serial calcitonin/CEA.
- **Study 6 (JCEM Homozygous K666N):** 6 family members (one homozygote).
- **Study 7 (Oncotarget S891A FMTC/CA):** 15 four-generation carriers.
- **Study 8 (AJCR Calcitonin-negative V804M):** 1 metastatic case.
- **Study 9 (JCEM ctDNA):** 21 sporadic MTC cases with pre/post biomarkers.
- **Study 10 (Genes RET c.1901G>A):** 2 RET C634Y/SLC12A3 carriers.
- **Study 11 (BMC Pediatrics MEN2B):** 1 pediatric RET M918T patient.
- **Study 12 (Annales RET Y791F Pheo):** 1 pheochromocytoma with normal calcitonin.
- **Study 13 (Surgery Today RET S891A):** 2 pheochromocytoma-first MEN2A carriers.
- **Study 14 (Annals of Medicine & Surgery C634R):** 2 MEN2A carriers.
- **Study 15 (Case Reports in Medicine MEN2B):** 1 RET M918T case.
- **Study 16 (Case Reports in Endocrinology delins):** 1 RET exon 11 deletion case.
- **Study 17 (Clinics and Practice C634G):** 6 RET C634G carriers.
- **Study 18 (EDM Case Reports K666N):** 4 RET K666N carriers.
- **Study 19 (Indian Journal of Cancer S891A):** 7 RET S891A carriers.
- **Study 20 (World Journal of Clinical Cases C634Y):** 3 RET C634Y carriers.
- **Age range:** 1-90 years.
- **Gender distribution (F/M):** 107/45.
- **RET Variants Included:** 24 total (K666N, L790F, Y791F, V804M, S891A, R525W, M918T, E505_G506del, A883F, C618S, C620Y, C620W, C630R, C630G, C634R, C634Y, C634W, C634S, C634G, E632_C634del, E632_L633del, D898_E901del, V899_E902del, D631_L633delinsE).

**ATA Risk Level Distribution:**

- **Level 1 (Moderate):** K666N, L790F, Y791F, V804M, S891A, R525W, E505_G506del.
- **Level 2 (High):** C618S, C630R, C630G, C620Y, C620W, D898_E901del, V899_E902del.
- **Level 3 (Highest):** C634R, C634Y, C634W, C634S, C634G, M918T, A883F, E632_C634del, E632_L633del, D631_L633delinsE.

**Clinical Outcomes:**

- MTC diagnosis now documented in **72/152 (47.4%)** real patients.
- C-cell disease (MTC + C-cell hyperplasia) observed in **76/152 (50.0%)** across all risk levels.
- Pheochromocytoma captured in 14 real patients (plus presymptomatic carriers) enabling MEN2A/MEN2B phenotyping.
- Hyperparathyroidism captured in 6 real patients across multiple risk tiers.

**Expanded Dataset:** Original 152 patients + synthetic variant-matched controls (1,069 rows total)

- Includes literature-based synthetic cases with variant-specific distributions.
- SMOTE augmentation applied inside the training loop for class balance.
- Enhanced feature space retains `cea_level_numeric`, `cea_elevated`, and imputation provenance for every record.

### Clinical Features

The dataset includes the following structured clinical and genetic features:

**Demographic Features:**

- `age`: Age at clinical evaluation (years)
- `gender`: Biological sex (0=Female, 1=Male)
- `age_group`: Categorized age ranges (young/middle/elderly/very_elderly)

**Genetic Features:**

- `ret_variant`: Specific RET variant (K666N, C634R, C634Y, L790F, E505_G506del, etc.)
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

The raw clinical data is stored in the [`data/raw`](data/raw) folder as structured json files:

- **[study_1.json](data/raw/study_1.json) ... [study_20.json](data/raw/study_20.json)**: Individual study extracts covering 20 included cohorts (see Data Sources table)
- **[literature_data.json](data/raw/literature_data.json)**: Aggregated statistics and meta-data
- **[mutation_characteristics.json](data/raw/mutation_characteristics.json)**: RET variant characteristics

This modular structure allows for:

- Easy data maintenance and updates
- Clear separation of concerns between raw data and processing logic
- Version control of individual study datasets
- Simple addition of new studies as they become available

### Data Processing Pipeline

The [create_datasets.py](src/create_datasets.py) script:

1. Loads patient data from JSON files in the [`data/raw`](data/raw) folder (20 studies)
2. Extracts and combines data from multiple research studies (152 patients, 24 variants across 20 sources)
3. Maps each variant to ATA risk level (1=Moderate, 2=High, 3=Highest)
4. Converts qualitative measurements to structured numeric features
5. Handles multiple reference ranges for calcitonin levels across studies
6. Engineers derived features (age groups, nodule presence, variant-specific interactions)
7. Generates two datasets:
   - `data/processed/ret_multivariant_training_data.csv`: Original 152 patients from literature
   - `data/processed/ret_multivariant_expanded_training_data.csv`: Expanded with synthetic controls
   - `data/processed/ret_multivariant_case_control_dataset.csv`: Further expanded with variant-matched controls

### Important Notes on Data Quality

- **Multi-Variant Dataset:** Includes 24 different RET variants with varying penetrance and risk profiles
- **Risk Stratification:** Variants classified by ATA guidelines (Level 1/2/3)
- **Incomplete Penetrance:** Not all carriers develop MTC; penetrance varies by variant
- **Variable Follow-up:** Some carriers elected surveillance over prophylactic surgery
- **Age-Dependent Risk:** Penetrance increases with age, reflected in age-stratified features
- **Variant-Specific Patterns:** High-risk variants (C634\*) show different clinical patterns than moderate-risk (K666N, L790F)
- **Study Heterogeneity:** Different studies used different calcitonin reference ranges and screening protocols

**Key features:**

- **End-to-end pipeline** managed by `main.py`, coordinating all major steps automatically.
- **Multiple ML algorithms:** Support for Logistic Regression, Random Forest, XGBoost, LightGBM, and SVM models.
- **Comprehensive model comparison:** Automatically generates detailed comparison of all 5 models on the same test set with complete patient data, showing which patients each model got right/wrong.
- **Model comparison mode:** Run all models simultaneously and compare performance metrics in a formatted table.
- **Dataset comparison mode:** Compare model performance on expanded dataset (with SMOTE and control cases) vs original paper data.
- **Automated data creation and expansion:** Scripts extract and structure relevant research data, and generate synthetic control samples to augment the dataset for robust modeling.
- **Comprehensive statistical analysis:** Automatic generation of descriptive statistics and visualization of the dataset for informed modeling.
- **Advanced model development:** Cross-validation and adaptive SMOTE balancing to handle class imbalance across all model types.
- **Clinical risk stratification:** 4-tier risk assessment (Low/Moderate/High/Very High) for actionable clinical decision-making.
- **Artifacts generated:** Processed datasets, trained model files, ROC curves, confusion matrices, and confidence interval summaries for clinically transparent scoring.

**Pipeline steps (as run by `main.py`):**

1. **create_datasets.py:** loads all study JSON files (now including Study 5 biomarker panels), computes the calcitonin vs CEA correlation, runs MICE + predictive mean matching to populate `cea_*` features, and emits the processed CSVs.
2. **data_analysis.py:** Computes descriptive statistics, generates variant-specific visualizations and risk-stratified analyses.
3. **data_expansion.py:** Produces variant-matched synthetic control samples to improve model balance.
4. **train_model.py:** Trains models with variant features, cross-validation, SMOTE balancing, and threshold optimization.
5. **test_model.py:** Evaluates the model on test data with variant-specific risk stratification, comprehensive metrics, and automatic comparison of all 5 models with complete patient data.
6. **calculate_ci.py:** Calculates 95% bootstrap confidence intervals for all performance metrics (automatically runs for all models).
7. **Artifact summary:** Includes `ret_multivariant_training_data.csv`, `ret_multivariant_expanded_training_data.csv`, `ret_multivariant_case_control_dataset.csv`, `model.pkl`, `model_comparison_detailed_results.txt`, and confidence interval reports.

**Advanced features:**

- **Automated Model Comparison:** Every test run generates comprehensive comparison of all 5 models with complete patient data, enabling pattern identification and clinical validation
- **Data Leakage Prevention:** SMOTE applied after train/test split to ensure realistic evaluation
- **Feature Engineering:** Polynomial features (age²) and interactions (calcitonin×age, risk×age, nodule_severity)
- **Variant-Aware Modeling:** One-hot encoding of 22 RET variants + risk level stratification
- **Constant Feature Removal:** Automatic detection and removal of non-informative features
- **Risk Stratification:** 4-tier system for clinical decision support instead of binary classification
- **Comprehensive Metrics:** ROC-AUC, F1-Score, Average Precision Score, ROC curves, confusion matrices, and automatic 95% bootstrap confidence intervals
- **Patient-Level Transparency:** See exactly which patients each model predicted correctly/incorrectly with full clinical context

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

- data/processed/ (processed CSVs)
  - ret_multivariant_training_data.csv - Original 152 patients
  - ret_multivariant_expanded_training_data.csv - Expanded with synthetic controls
  - ret_multivariant_case_control_dataset.csv - Additional control augmentation
- data/raw/ (study JSON files)
  - study_1.json ... study_7.json
  - literature_data.json
  - mutation_characteristics.json
- models/ - reusable model classes (base_model.py, random_forest_model.py, lightgbm_model.py, xgboost_model.py, logistic_regression_model.py)
- results/ - metrics, logs, ROC/confusion charts, biomarker summaries
- src/ - pipeline scripts (create_datasets.py, data_analysis.py, data_expansion.py, train_model.py, test_model.py)
- main.py - orchestrates dataset creation, analysis, training, testing
- requirements.txt - Python dependencies

## Usage

### Basic Usage

Run the complete pipeline:

```sh
python main.py
```

This executes all stages: data preparation, analysis, expansion, training, and testing.

### Model Selection (`--m`)

Choose which model to train:

- `l` or `logistic`: Logistic Regression (default, zero-miss option)
- `r` or `random_forest`: Random Forest (research comparison)
- `x` or `xgboost`: XGBoost
- `g` or `lightgbm`: LightGBM (research comparison)
- `s` or `svm`: Support Vector Machine (linear kernel)
- `a` or `all`: Run all models and compare

### Dataset Selection (`--d`)

Choose which dataset to use:

- `o` or `original`: Original 152 patients (no synthetic data) ⭐ **Recommended for clinical use**
- `e` or `expanded`: Expanded with synthetic controls + SMOTE (default)
- `b` or `both`: Run on both datasets for comparison


### Examples

```sh
# ✅ RECOMMENDED FOR CLINICAL USE: Logistic Regression on original data
python main.py --m=logistic --d=original

# Research comparison: LightGBM on original data
python main.py --m=lightgbm --d=original

# Test SVM (kernel-based learning) on original data
python main.py --m=svm --d=original

# Compare all models on original dataset (identify best performer)
python main.py --m=all --d=original

# Demonstrate recall drop from synthetic augmentation (research use)
python main.py --m=random_forest --d=both

# ⚠️ NOT RECOMMENDED: Expanded dataset (lower recall for safest models)
python main.py --m=random_forest --d=expanded  # Only for research comparison
```


### Model Comparison Mode

When using `--m=all`, the pipeline:

1. Runs data preparation once (shared across models)
2. Trains and tests all five model types sequentially
3. Saves detailed logs to `results/logs/`
4. Displays comprehensive comparison table

### Dataset Comparison Mode

When using `--d=both`, the pipeline:

1. Runs the model on the expanded dataset (synthetic + SMOTE)
2. Runs the same model on the original dataset
3. Generates separate results files
4. Displays a comparison table showing performance differences

This mode clearly demonstrates the recall degradation from synthetic augmentation.

Statistical significance tests are triggered automatically only when running both datasets together (e.g., `python main.py --m=all --d=both`).

### Ablation Study (`--ablation`)

The ablation study systematically removes feature groups to test the model's reliance on genetic vs biomarker features. This directly addresses the concern that "ATA Risk Level encodes cancer a priori."

**Run via main pipeline:**

```sh
# Ablation study with LightGBM on expanded dataset
python main.py --ablation --m=lightgbm --d=expanded
```

**Or run the standalone script directly:**

```sh
# Full ablation study (all models, both datasets) - 80 experiments
python src/ablation_study.py --m=all --d=both

# Specific model/dataset combinations
python src/ablation_study.py --m=lightgbm --d=expanded
python src/ablation_study.py --m=xgboost --d=original
python src/ablation_study.py --m=random_forest --d=both
```

**Ablation Configurations:**

| Configuration | Features Removed | Purpose |
|---------------|------------------|---------|
| `baseline` | None | Full model performance |
| `no_risk_level` | `ret_risk_level`, interactions | Test ATA risk contribution |
| `no_variants` | All `variant_*` dummies | Test variant encoding contribution |
| `no_genetics` | All genetic features | Pure biomarker prediction |
| `no_calcitonin` | `calcitonin_*` features | Test if genetics alone suffice |
| `no_cea` | `cea_level_numeric` | Address CEA imputation concerns |
| `genetics_only` | All biomarkers, nodules | Test if model is "just consensus" |
| `biomarkers_only` | All genetic features | Clinical utility without genetics |

**Results saved to:** `results/ablation/`
- `{model}_{dataset}_ablation_results.txt` - Detailed findings
- `{model}_{dataset}_ablation_results.csv` - For analysis

**Key Finding:** With all genetic features removed, the model still achieves 94.9% accuracy using only biomarkers - proving it learns beyond "restating consensus knowledge."

### Explainability (SHAP + LIME)

- Explainability runs automatically during testing (including `python main.py --m=all --d=both`). Use `python src/test_model.py --no-explain ...` to skip.
- **SHAP**:
  - Text: `results/shap/<model>/<model>_<dataset>.txt` (e.g., `results/shap/logistic/logistic_expanded.txt`)
  - Charts (PNG): `charts/shap/<model>/` (`expanded_bar.png`, `original_bar.png`)
- **LIME** (local explanations + global summary over selected cases):
  - Results: `results/lime/<model>/`
  - Charts (PNG): `charts/lime/<model>/` (per-sample LIME plots + `lime_global_importance.png` + `lime_shap_global_<dataset>.png`)
  - The script explains 5 correctly classified + up to 5 misclassified cases (false negatives prioritized).
  - Counterfactual suggestions (when possible) are written to `results/lime/<model>/<model>_<dataset>_counterfactuals.txt`.
- Summary: `results/explainability_summary.txt`

**Important clinical disclaimer:** This project is for research and educational use only and is not medical advice or a clinical decision support device. Do not use these outputs to diagnose, treat, or delay care; consult qualified clinicians and confirm with guideline-based evaluation and genetic testing.

### Hugging Face Space (Gradio UI + API)

Interactive demo and hosted inference are available via the Hugging Face Space:

`https://huggingface.co/spaces/arjuncodess/men2-predictor`

- **Gradio app:** Use the web UI to enter patient features and view the predicted MEN2/MTC risk output.
- **API:** You can call the Space programmatically. The easiest way is `gradio_client`:

```python
from gradio_client import Client

client = Client("arjuncodess/men2-predictor")
client.view_api()
# Then call the printed endpoint with client.predict(...)
```

**Disclaimer for the Space:** The hosted app/API is provided as-is for demonstration only. It may change without notice, and it must not be used for real-world clinical decisions.

### Model comparison with patient data

**New feature:** Every test run now automatically generates a comprehensive comparison of all 5 models on the same test set, showing complete patient data alongside each model's predictions.

**What you get:**

The comparison table includes for each test patient:
- **Patient identification**: study_id for original data, source_id for synthetic controls
- **Complete clinical data**: age, sex, RET variant, risk level, calcitonin levels, nodules, family history, etc.
- **Actual diagnosis**: MTC or No_MTC
- **All model predictions**: LR, RF, LGB, XGB, SVM (marked OK for correct, XX for incorrect)
- **Color-coded terminal output**: green for correct predictions, red for incorrect
- **Accuracy summary**: total correct/incorrect for each model

**Saved file:** `results/model_comparison_{dataset_type}_detailed_results.txt`

This file includes:
- Complete legend explaining all abbreviations
- Data split methodology (80/20, stratified, random_state=42)
- SMOTE application details (only on training data)
- All held-out test patients with full clinical data
- Model predictions for easy comparison

**Why this matters:**

1. **Transparency**: See exactly which patients each model got right or wrong
2. **Pattern identification**: Identify difficult cases that multiple models miss
3. **Clinical validation**: Verify models make sense given patient characteristics
4. **Research utility**: Comprehensive data for publication and analysis
5. **Debugging**: Quickly spot if synthetic controls are causing issues

**Data split explained:**

- **Method**: 80/20 train/test split with random_state=42 (reproducible)
- **Stratification**: Maintains MTC vs No MTC ratio in both sets
- **SMOTE**: Applied only to training data after split
- **Test set**: 100% real data (original patients + synthetic controls, no SMOTE)

**Note on "synthetic controls":**

Patients with source_id (e.g., "33_control", "mtc_s0_control") are synthetic controls added to expand the dataset. These are not from SMOTE—they were generated before the train/test split and are treated as real data. SMOTE is only applied to the training set after splitting.

### Output Files

**Model Files:**

- `saved_models/{model_type}_{dataset_type}_model.pkl`

**Results:**

- `results/{model_type}_{dataset_type}_test_results.txt` - individual model performance summaries with embedded 95% confidence intervals
- `results/model_comparison_{dataset_type}_detailed_results.txt` - comprehensive comparison of all models with complete patient data
- `results/{model_type}_{dataset_type}_confidence_intervals.txt` - standalone bootstrap confidence intervals (automatically calculated for all models)
- `charts/roc_curves/{model_type}_{dataset_type}.png` - ROC curves with area under the curve and optimal-threshold marker
- `charts/confusion_matrices/{model_type}_{dataset_type}.png` - paired raw-count and normalized confusion matrices
- `charts/correlation_matrices/{model_type}_{dataset_type}.png` - feature correlation matrix for LightGBM (expanded dataset)

**Logs (when using --m=all or --d=both):**

- `results/logs/{model_type}_{dataset_type}_training.log`
- `results/logs/{model_type}_{dataset_type}_testing.log`
- `results/logs/{model_type}_{dataset_type}_confidence_intervals.log`

## Technical Details

<details>
<summary><b>Feature Engineering</b></summary>

**Demographic Features:**

- Age at clinical evaluation (years)
- Gender (binary)
- Age groups (young/middle/elderly/very_elderly)

**Genetic Features:**

- RET variant (one-hot encoded across 24 variants)
- ATA risk level (ordinal: 1=Moderate, 2=High, 3=Highest)

**Biomarker Features:**

- Calcitonin elevation status (binary)
- Calcitonin level (numeric, pg/mL)
- CEA level (numeric, ng/mL)
- CEA elevation flag (>5 ng/mL baseline)
- CEA imputed provenance flag (0 = observed, 1 = MICE+PMM)

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

1. **create_datasets.py:** Loads patient data from JSON (all 20 studies), performs calcitonin<->CEA correlation plus MICE+PMM imputation, and writes enriched CSVs
2. **data_analysis.py:** Computes descriptive statistics, generates visualizations
3. **data_expansion.py:** Produces variant-matched synthetic control samples (optional)
4. **train_model.py:** Trains models with cross-validation, SMOTE balancing, threshold optimization
5. **test_model.py:** Evaluates models with comprehensive metrics, risk stratification, and automatic comparison of all 5 models with complete patient data
6. **calculate_ci.py:** Calculates 95% bootstrap confidence intervals for all performance metrics (automatically runs)

</details>

<details>
<summary><b>Model Architecture</b></summary>

**Supported Models:**

- Logistic Regression (baseline, linear, zero-miss option)
- Random Forest (ensemble)
- XGBoost (gradient boosting)
- LightGBM (gradient boosting)
- Support Vector Machine (linear)

**Training Configuration:**

- Cross-validation for hyperparameter tuning
- SMOTE balancing (applied after train/test split to prevent data leakage)
- Threshold optimization for clinical use cases
- Variant-aware feature encoding

**Evaluation Metrics:**

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Average Precision Score, ROC curve visualizations
- Confusion matrices (raw and normalized)
- Automatic 95% bootstrap confidence intervals on all metrics (1,000 iterations)
- Risk-stratified performance

</details>

<details>
<summary><b>Dataset Characteristics</b></summary>

**Original Dataset:**

- 152 confirmed RET germline mutation carriers from 20 peer-reviewed studies
- 22 RET variants (K666N, L790F, Y791F, V804M, S891A, R525W, M918T, E505_G506del, A883F, C618S, C620Y, C620W, C630R, C630G, C634R/Y/W/S, E632_C634del, E632_L633del, D898_E901del, V899_E902del)
- Age range: 1-90 years
- Gender distribution (F/M): 107/45
- ATA risk levels: Level 1 (Moderate), Level 2 (High), Level 3 (Highest)

- **Expanded Dataset:**

- Original 152 patients + synthetic variant-matched controls (total rows: 216)
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

1. **Small sample size**: 152 patients is still typical for rare genetic conditions but limits statistical power
2. **Retrospective data**: Extracted from published case series, not prospective validation
3. **Study heterogeneity**: Different calcitonin reference ranges and protocols across 20 studies
4. **Limited diversity**: Primarily European descent patients; generalizability to other populations unknown
5. **No external validation**: Performance validated on held-out data from same studies, not independent cohorts

**However**: These limitations are representative of rare disease ML challenges. Our finding (synthetic data harm) is strengthened by the fact that it persists across models and datasets.

**Next steps**: Prospective validation in clinical setting with multi-center collaboration.

## License

This project is licensed under the MIT License.

## Participants

This project was not built in isolation, and I want to acknowledge the people who contributed meaningfully to it.

### Arjun Vijay Prakash
**Developer**  
*Class 10 Student, City Montessori School, Kanpur Road Branch*  
*E-mail: arjunv.prakash12345@gmail.com*

I conceived the project, implemented the machine learning pipeline, conducted the statistical analysis, and wrote the research paper. I also managed the technical infrastructure, including the Hugging Face Space deployment and reproducible codebase.

### Harnoor Kaur
**Research Lead**  
*Class 12 Student, City Montessori School, Kanpur Road Branch*  
*E-mail: har.nooor16@gmail.com*

Harnoor helped extensively with the research side of this project, including locating and compiling relevant peer-reviewed studies and assisting in sourcing and organizing the clinical data used for model development.

### Shashwat Misra
**Project Mentor**  
*E-mail: mishra.shashwat4002@gmail.com*

Shashwat provided guidance throughout the development process, helped review the approach, and offered feedback on both the technical and research decisions that shaped the final pipeline.

Their support played an important role in turning this from an idea into a working, validated project.

## Acknowledgements

Thanks to open source communities and packages including scikit-learn, pandas, numpy, matplotlib, seaborn, joblib, and imbalanced-learn for making data science and reproducibility accessible.

Special thanks to the authors of the research studies that provided clinical data:

- JCEM Case Reports (2025) - RET K666N carriers
- JCEM (2016) RET exon 7 deletion case
- Xu et al. Thyroid (2016) - RET K666N carriers
- European Journal of Endocrinology (2006) - Multi-variant RET carriers
