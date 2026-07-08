"""Leave-one-study-out grouped validation (sensitivity analysis).

The primary manuscript benchmark uses a row-wise stratified 80/20 split. Because
the 149 records come from only 10 studies and several sources are single
families/pedigrees, a row-wise split can place related carriers in both train
and test, which may make internal performance look optimistic.

This script re-evaluates the original 149-record cohort with grouped
cross-validation keyed on ``study_id`` (Leave-One-Study-Out), so every record
from a given source study stays entirely in either train or test. ``study_id``
is the best available proxy for family/pedigree grouping since clean family IDs
are not recoverable for all sources (studies 4-8 and 10 are single families).

All preprocessing (CEA imputation, scaling, SMOTE) is fit on training folds
only, matching the leakage controls used elsewhere. Both the genotype-blind
(triage) and genotype-aware (sequencing-informed) feature sets are evaluated so
the grouped result can be compared directly with the 80/20 split.
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.dirname(__file__))
from preprocessing import impute_cea_train_only, fill_remaining_na_train_only
from reporting_metrics import metrics_from_counts, format_metric
from ablation_study import prepare_features, apply_ablation, get_model, ABLATION_CONFIGS


# Feature tracks mapped onto existing ablation configs for consistency.
TRACKS = {
    "genotype_blind": {
        "name": "Genotype-blind (triage)",
        "config": "no_genetics",
    },
    "genotype_aware": {
        "name": "Genotype-aware (sequencing-informed)",
        "config": "baseline",
    },
}


def run_track(features, target, groups, config_name, model_type):
    """Run Leave-One-Study-Out CV for one feature track; pool OOF predictions."""
    ablated = apply_ablation(features, config_name)
    logo = LeaveOneGroupOut()

    oof_true = []
    oof_pred = []
    oof_proba = []
    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(ablated, target, groups)):
        X_train = ablated.iloc[train_idx]
        X_test = ablated.iloc[test_idx]
        y_train = target.iloc[train_idx]
        y_test = target.iloc[test_idx]
        held_out_study = groups.iloc[test_idx].iloc[0]

        # Training-only CEA imputation + residual NaN fill.
        X_train, X_test, _ = impute_cea_train_only(X_train, X_test, random_state=42)
        X_train, X_test = fill_remaining_na_train_only(X_train, X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # SMOTE on training fold only, guarded for tiny minority classes.
        y_counts = Counter(y_train)
        minority = min(y_counts.values()) if len(y_counts) > 1 else 0
        if len(y_counts) > 1 and minority > 1:
            k = min(5, minority - 1)
            X_train_scaled, y_train = SMOTE(random_state=42, k_neighbors=k).fit_resample(
                X_train_scaled, y_train
            )

        model = get_model(model_type)
        model.train(X_train_scaled, y_train, scaler, ablated.columns.tolist())

        y_pred = model.model.predict(X_test_scaled)
        y_proba = model.model.predict_proba(X_test_scaled)[:, 1]

        oof_true.extend(y_test.tolist())
        oof_pred.extend(np.asarray(y_pred).tolist())
        oof_proba.extend(np.asarray(y_proba).tolist())

        fold_pos = int((y_test == 1).sum())
        fold_correct = int((np.asarray(y_pred) == y_test.values).sum())
        fold_rows.append({
            "fold": fold_idx + 1,
            "held_out_study": held_out_study,
            "n": len(y_test),
            "n_mtc": fold_pos,
            "correct": fold_correct,
        })

    oof_true = np.asarray(oof_true)
    oof_pred = np.asarray(oof_pred)
    oof_proba = np.asarray(oof_proba)

    tp = int(((oof_true == 1) & (oof_pred == 1)).sum())
    tn = int(((oof_true == 0) & (oof_pred == 0)).sum())
    fp = int(((oof_true == 0) & (oof_pred == 1)).sum())
    fn = int(((oof_true == 1) & (oof_pred == 0)).sum())

    pooled = metrics_from_counts(tn, fp, fn, tp, y_true=oof_true, y_score=oof_proba)
    return pooled, fold_rows


def run_grouped_validation(model_type="xgboost"):
    df = pd.read_csv("data/processed/ret_multivariant_training_data.csv")
    features, target = prepare_features(df, target_column="mtc_diagnosis")

    if "study_id" not in df.columns:
        raise ValueError("study_id column required for grouped validation")
    groups = df.loc[features.index, "study_id"] if len(features) == len(df) else df["study_id"]
    groups = groups.reset_index(drop=True)
    features = features.reset_index(drop=True)
    target = target.reset_index(drop=True)

    n_groups = groups.nunique()
    print("=" * 80)
    print("LEAVE-ONE-STUDY-OUT GROUPED VALIDATION (SENSITIVITY ANALYSIS)")
    print("=" * 80)
    print(f"Model: {model_type.upper()} | Records: {len(df)} | Studies (groups): {n_groups}")
    print(f"Target distribution: {target.value_counts().to_dict()}\n")

    os.makedirs("results/grouped_validation", exist_ok=True)
    out_path = f"results/grouped_validation/{model_type}_leave_one_study_out.txt"

    track_results = {}
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("LEAVE-ONE-STUDY-OUT GROUPED VALIDATION (SENSITIVITY ANALYSIS)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {model_type.upper()}\n")
        f.write(f"Records: {len(df)} | Studies (groups): {n_groups}\n")
        f.write("Grouping: study_id (proxy for family/pedigree). All records from a\n")
        f.write("study stay entirely in train or test. Preprocessing is train-fold only.\n")
        f.write("Metrics are pooled out-of-fold: point% (num/den; Wilson 95% low%-high%).\n\n")

        for track_key, track in TRACKS.items():
            pooled, fold_rows = run_track(
                features, target, groups, track["config"], model_type
            )
            track_results[track_key] = pooled

            print(f"[{track['name']}]")
            c = pooled["counts"]
            print(f"  CM: TN={c['tn']} FP={c['fp']} FN={c['fn']} TP={c['tp']}")
            print(f"  Sensitivity: {format_metric(pooled['sensitivity'])}")
            print(f"  Specificity: {format_metric(pooled['specificity'])}")
            print(f"  Accuracy:    {format_metric(pooled['accuracy'])}")
            roc = f"{pooled['roc_auc']:.3f}" if pooled["roc_auc"] is not None else "n/a"
            print(f"  ROC-AUC:     {roc}\n")

            f.write("-" * 80 + "\n")
            f.write(f"{track['name']} (config: {track['config']})\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Confusion matrix (pooled OOF): TN={c['tn']}, FP={c['fp']}, "
                    f"FN={c['fn']}, TP={c['tp']}\n")
            f.write(f"  Sensitivity: {format_metric(pooled['sensitivity'])}\n")
            f.write(f"  Specificity: {format_metric(pooled['specificity'])}\n")
            f.write(f"  Accuracy:    {format_metric(pooled['accuracy'])}\n")
            f.write(f"  ROC-AUC:     {roc}\n")
            f.write("  Per-fold (held-out study):\n")
            for row in fold_rows:
                f.write(f"    Fold {row['fold']:>2} | study={row['held_out_study']:<9} | "
                        f"n={row['n']:>3} | MTC={row['n_mtc']:>3} | "
                        f"correct={row['correct']:>3}/{row['n']}\n")
            f.write("\n")

    print(f"Results saved to: {out_path}")
    return track_results


def main():
    parser = argparse.ArgumentParser(description="Leave-one-study-out grouped validation")
    parser.add_argument("--m", "--model", type=str, default="xgboost",
                        choices=["logistic", "random_forest", "xgboost", "lightgbm", "svm"],
                        help="Model type (default: xgboost)")
    args = parser.parse_args()
    run_grouped_validation(args.m)


if __name__ == "__main__":
    main()
