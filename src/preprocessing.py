"""Shared preprocessing helpers.

The key purpose of this module is to keep CEA imputation strictly training-only.
Fitting MICE (IterativeImputer) on the full cohort before the train/test split
leaks held-out test information into the training rows and vice versa. To avoid
that, the raw datasets now carry observed-or-NaN CEA values, and imputation is
performed here after every split/fold using an imputer fit on training rows only.
"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

CEA_COLUMN = "cea_level_numeric"
CALCITONIN_COLUMN = "calcitonin_level_numeric"


def _predict_cea(imputer, frame):
    """Return imputer-predicted CEA values for the provided rows."""
    transformed = imputer.transform(frame[[CALCITONIN_COLUMN, CEA_COLUMN]])
    return pd.Series(transformed[:, 1], index=frame.index)


def _pmm_match(predicted_values, donors, rng, n_neighbors=5):
    """Predictive mean matching: snap predictions to observed training donors."""
    donors = np.asarray([d for d in donors if d is not None], dtype=float)
    matched = []
    for value in predicted_values:
        distances = np.abs(donors - value)
        k = min(n_neighbors, len(donors))
        nearest = np.argsort(distances)[:k]
        choice = rng.choice(nearest)
        matched.append(float(max(0.0, donors[choice])))
    return matched


def fit_cea_imputer(X_train, random_state=42):
    """Fit an MICE+PMM CEA imputer using training rows only.

    Returns a state dict that can be applied to any frame with
    ``apply_cea_imputer``. If there are too few observed CEA donors to fit a
    regression, the state falls back to the training median (still train-only).
    """
    state = {"imputer": None, "donors": [], "cea_median": None, "fitted": False}

    if CEA_COLUMN not in X_train.columns or CALCITONIN_COLUMN not in X_train.columns:
        return state

    observed = X_train[X_train[CEA_COLUMN].notna()]
    state["donors"] = observed[CEA_COLUMN].tolist()

    if len(observed) >= 2 and observed[CEA_COLUMN].nunique() > 1:
        imputer = IterativeImputer(
            random_state=random_state,
            sample_posterior=True,
            max_iter=20,
            min_value=0.0,
            imputation_order="ascending",
        )
        imputer.fit(observed[[CALCITONIN_COLUMN, CEA_COLUMN]])
        state["imputer"] = imputer
        state["fitted"] = True

    if state["donors"]:
        state["cea_median"] = float(np.median(state["donors"]))

    return state


def apply_cea_imputer(X, state, random_state=42):
    """Return a copy of ``X`` with missing CEA filled from a train-fit state."""
    X = X.copy()
    if CEA_COLUMN not in X.columns:
        return X

    missing_mask = X[CEA_COLUMN].isna()
    if not missing_mask.any():
        return X

    rng = np.random.default_rng(random_state)
    if state.get("fitted") and state.get("donors"):
        predicted = _predict_cea(state["imputer"], X.loc[missing_mask])
        matched = _pmm_match(predicted.values, state["donors"], rng)
        X.loc[missing_mask, CEA_COLUMN] = matched
    elif state.get("cea_median") is not None:
        X.loc[missing_mask, CEA_COLUMN] = state["cea_median"]
    else:
        X.loc[missing_mask, CEA_COLUMN] = 0.0

    X[CEA_COLUMN] = X[CEA_COLUMN].clip(lower=0.0)
    return X


def impute_cea_train_only(X_train, X_test, random_state=42):
    """Fit the CEA imputer on ``X_train`` and apply it to both frames."""
    state = fit_cea_imputer(X_train, random_state=random_state)
    X_train_imputed = apply_cea_imputer(X_train, state, random_state=random_state)
    X_test_imputed = apply_cea_imputer(X_test, state, random_state=random_state)
    return X_train_imputed, X_test_imputed, state


def fill_remaining_na_train_only(X_train, X_test):
    """Fill any residual NaNs using training medians only (no test leakage)."""
    medians = X_train.median(numeric_only=True)
    X_train_filled = X_train.fillna(medians).fillna(0.0)
    X_test_filled = X_test.fillna(medians).fillna(0.0)
    return X_train_filled, X_test_filled
