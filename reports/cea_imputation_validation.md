# CEA Imputation Validation Study Report

## Response to Reviewer Concerns Regarding CEA Imputation

---

## Executive Summary

This report presents findings from a comprehensive sensitivity analysis conducted to address reviewer concerns that the weak calcitonin-CEA correlation (r=0.1158, n=12) may undermine the reliability of CEA imputation in our MEN2 prediction model.

The main human framing is: **Can we save those 20k Rs people with just a simple blood test?** In India, genetic testing for MEN2 costs INR 20,000 (~$225 USD), putting life-saving diagnosis out of reach for most families. This validation study tests the biomarker part of that question by asking whether CEA meaningfully helps routine blood-marker modeling.

**Key Finding:** CEA contribution is model-dependent. For **LightGBM on the expanded dataset**, CEA improves accuracy from **92.86%** to **96.19%**, and **MICE+PMM** is the strongest imputation strategy. For **XGBoost on the original dataset**, removing CEA improves accuracy from **83.33%** to **90.00%** while preserving **100.00% recall**.

---

## Reviewer Concern

> *"The calcitonin-CEA correlation is weak (r=0.24, n=34). This modest correlation raises concerns about the reliability of MICE+PMM imputation for CEA values, which may introduce noise or bias into the prediction model."*

---

## Methodology

### Study Design

We conducted a two-part sensitivity analysis across:
- **5 machine learning models:** Logistic Regression, Random Forest, XGBoost, LightGBM, SVM
- **2 datasets:** Original (149 patients) and Expanded (1,047 samples with synthetic augmentation)
- **2 study options:** CEA presence/absence and imputation method comparison

### Option A: CEA Presence/Absence Comparison

Compare model performance WITH vs WITHOUT CEA features to quantify CEA's actual contribution to prediction.

### Option B: Imputation Method Comparison

Test 5 different imputation strategies to assess robustness:

| Method | Description |
|--------|-------------|
| MICE+PMM | Multiple Imputation by Chained Equations with Predictive Mean Matching |
| Mean Imputation | Replace missing CEA with mean of observed values |
| Median Imputation | Replace missing CEA with median of observed values |
| Zero Imputation | Replace missing CEA with zero (conservative lower bound) |
| Complete Case | Use only patients with observed CEA values |

### Statistical Approach

- 80/20 stratified train-test split with fixed random seed (42)
- SMOTE applied only to training data to prevent data leakage
- StandardScaler fitted only on training data
- Consistent evaluation metrics: Accuracy, Recall, F1, ROC-AUC

---

## Results

### Finding 1: CEA Effects Depend on the Model

The effect of CEA differs in the most relevant original-data and expanded-data configurations:

| Model | Dataset | With CEA | Without CEA | Accuracy Change |
|-------|---------|----------|-------------|-----------------|
| **XGBoost** | Original | 83.33% | **90.00%** | **+6.67% without CEA** |
| **LightGBM** | Expanded | **96.19%** | 92.86% | **-3.33% without CEA** |

**Interpretation:** CEA is not universally beneficial or universally unnecessary. It is dispensable for the primary original-data XGBoost model, but helpful for the highest-accuracy LightGBM model.

---

### Finding 2: Imputation Method Choice Affects Accuracy-Recall Tradeoffs

The two most relevant models respond differently to imputation:

| Method | XGBoost Original Accuracy / Recall | LightGBM Expanded Accuracy / Recall |
|--------|------------------------------------|-------------------------------------|
| MICE+PMM (Current) | 83.33% / **100.00%** | **96.19% / 90.20%** |
| Mean Imputation | 86.67% / 93.33% | 93.81% / 86.27% |
| Median Imputation | 86.67% / 93.33% | 92.86% / 90.20% |
| Zero Imputation | 86.67% / 93.33% | 93.81% / 86.27% |
| Complete Case | 33.33% / 50.00% | 69.23% / 66.67% |

**Interpretation:** For LightGBM on the expanded dataset, MICE+PMM is the best overall option. For XGBoost on the original dataset, MICE+PMM preserves perfect recall, while simpler imputations raise accuracy slightly at the cost of sensitivity.

---

### Finding 3: Sensitivity and Accuracy Favor Different CEA Strategies

The original-data sensitivity benchmark prioritizes recall, while the expanded simulation emphasizes overall discrimination:

| Model | Best Recall Setting | Best Accuracy Setting |
|-------------------|-----------------|----------|
| **XGBoost (Original)** | 100.00% recall with CEA or without CEA | 90.00% accuracy without CEA |
| **LightGBM (Expanded)** | 90.20% recall with CEA or median imputation | 96.19% accuracy with MICE+PMM |

**Interpretation:** The preferred CEA strategy depends on the analysis goal. For the primary original-data XGBoost model, CEA is not required. For the highest-accuracy expanded LightGBM model, CEA is helpful.

---

## Cross-Model Consistency

The findings are directionally consistent across the broader benchmark in showing that CEA effects are modest relative to the main predictive structure, but the sign of the effect is not identical:

| Model | Dataset | With CEA | Without CEA | Delta Accuracy |
|-------|---------|----------|-------------|----------------|
| Logistic Regression | Original | 66.67% | 73.33% | +6.67% |
| Random Forest | Original | 83.33% | 73.33% | -10.00% |
| **XGBoost** | Original | 83.33% | **90.00%** | +6.67% |
| **LightGBM** | Expanded | **96.19%** | 92.86% | -3.33% |
| SVM | Expanded | 85.71% | 89.52% | +3.81% |

**Pattern Summary:**

| Finding | XGBoost Original | LightGBM Expanded |
|---------|------------------|-------------------|
| Best use case | Original-data sensitivity | Expanded-data accuracy |
| CEA required? | No | Yes, helpful |
| Best imputation by accuracy | No CEA | MICE+PMM |
| Best imputation by recall | With or without CEA | MICE+PMM or median |

---

## Response to Reviewer Critique

### Claim: "Weak CEA correlation undermines imputation reliability"

**Finding:** The weak correlation does not invalidate the model. Instead, it means CEA must be interpreted in a model-specific way. In XGBoost on the original dataset, CEA can be omitted without harming recall. In LightGBM on the expanded dataset, CEA improves overall performance and MICE+PMM is the preferred strategy.

### Claim: "MICE+PMM may introduce noise or bias"

**Finding:** Imputation effects are measurable but not catastrophic. For LightGBM-expanded, MICE+PMM gives the best overall result. For XGBoost-original, simpler imputations trade a small gain in accuracy for lower recall.

### Clinical Rationale for Including CEA

CEA remains clinically relevant even when not required for the strongest original-data sensitivity configuration:

**1. Calcitonin Has Specificity Limitations**

Elevated calcitonin levels can occur in many conditions other than MTC:
- Hypercalcemia and hypergastrinemia
- Other neuroendocrine tumors
- Kidney insufficiency
- Papillary and follicular thyroid carcinomas
- Goiter and chronic autoimmune thyroiditis
- Medications (omeprazole, beta-blockers)

**2. CEA Adds Complementary Clinical Value**

- **Prognostic value:** CEA doubling time helps assess disease aggressiveness
- **Detection of aggressive MTC:** Rising CEA without calcitonin change may indicate poorly differentiated MTC
- **Clinical guidelines recommend both:** Current practice measures both markers for comprehensive MTC evaluation

**3. Combined Use is Clinical Standard of Care**

Clinical guidelines recommend measuring both serum calcitonin AND CEA together because both markers' doubling times serve as powerful predictors of recurrence and mortality. Including CEA aligns with standard clinical practice for MEN2 management.

---

## Methodological Validity Assessment

| Concern | Our Implementation | Verdict |
|---------|-------------------|---------| 
| Data leakage in splits | Train-test split before any processing | Valid |
| Feature scaling leakage | Scaler fitted only on training data | Valid |
| SMOTE applied correctly | Applied after split, training only | Valid |
| Multiple imputation methods | 5 strategies tested systematically | Valid |
| Multiple model validation | 5 diverse algorithms tested | Valid |
| Reproducibility | Fixed random seed (42) | Valid |

**Conclusion:** The validation study methodology is sound and results are reproducible.

---

## Conclusions

1. **CEA effects are model-dependent.** CEA improves the highest-accuracy LightGBM model but is not required for the primary original-data XGBoost model.

2. **MICE+PMM remains the preferred imputation strategy for LightGBM on the expanded dataset.** It delivers the strongest overall accuracy and F1 performance.

3. **The weak calcitonin-CEA correlation does not compromise validity.** It changes how CEA should be interpreted, but does not invalidate the models.

4. **Original-data sensitivity is maintained in the preferred benchmark model.** XGBoost on the original dataset preserves 100% recall with or without CEA.

5. **Results remain biologically interpretable across algorithms,** but the strongest benchmark conclusions come from XGBoost-original for original-data sensitivity and LightGBM-expanded for simulated accuracy.

6. **CEA inclusion remains clinically justified** because calcitonin alone has specificity limitations and combined assessment is clinical standard of care.

---

## Appendix: Detailed Results

### XGBoost (Original Dataset) - Option A

| Configuration | Accuracy | Recall | F1 Score | ROC-AUC |
|---------------|----------|--------|----------|---------|
| With CEA Features | 83.33% | 100.00% | 0.8571 | 0.9111 |
| Without CEA Features | 90.00% | 100.00% | 0.9091 | 0.9378 |

### LightGBM (Expanded Dataset) - Option B

| Imputation Method | Accuracy | Recall | F1 Score | ROC-AUC |
|-------------------|----------|--------|----------|---------|
| MICE+PMM (Current) | 96.19% | 90.20% | 0.9200 | 0.9908 |
| Mean Imputation | 93.81% | 86.27% | 0.8713 | 0.9819 |
| Median Imputation | 92.86% | 90.20% | 0.8598 | 0.9817 |
| Zero Imputation | 93.81% | 86.27% | 0.8713 | 0.9801 |
| Complete Case | 69.23% | 66.67% | 0.6667 | 0.7857 |

### Study Files

Complete results for all 10 model-dataset combinations are available in:
- `results/cea_validation/{model}_{dataset}_cea_validation.txt`
- `results/cea_validation/{model}_{dataset}_option_a.csv`
- `results/cea_validation/{model}_{dataset}_option_b.csv`

---

*Methodology: Systematic imputation method comparison with consistent train-test protocols.*
