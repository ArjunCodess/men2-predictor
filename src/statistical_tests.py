"""
Statistical significance tests comparing recall between original and expanded datasets.

The script parses the detailed per-patient prediction tables that are already generated
by ``src/test_model.py`` for both datasets. It then evaluates whether the drop in recall
observed after adding synthetic controls is statistically significant.

Methods implemented:
    * McNemar's test (only when overlapping positive patients exist, otherwise skipped)
    * A permutation test on the difference in recall between datasets
    * Bootstrap confidence intervals for the recall drop (original - expanded)
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List

import numpy as np
from scipy import stats


MODEL_ABBREVIATIONS = {
    "LR": "logistic",
    "RF": "random_forest",
    "LGB": "lightgbm",
    "XGB": "xgboost",
    "SVM": "svm",
}

MODEL_NAMES = {
    "logistic": "Logistic Regression",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "svm": "Support Vector Machine",
}

ALPHAS = (0.05, 0.01, 0.001)


@dataclass
class DatasetEntry:
    """Container for a single patient row extracted from the comparison tables."""

    patient_key: str
    actual_label: str
    predictions: Dict[str, str]


def _normalize_patient_key(text: str) -> str:
    """Normalize spacing so duplicate patients can be detected across datasets."""

    return " ".join(text.split())


def parse_prediction_table(dataset_type: str) -> List[DatasetEntry]:
    """
    Parse the detailed per-patient comparison file for the requested dataset.

    Parameters
    ----------
    dataset_type:
        Either ``"original"`` or ``"expanded"``.
    """

    file_path = (
        Path("results") / f"model_comparison_{dataset_type}_detailed_results.txt"
    )
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} not found. Run `python src/test_model.py --d={dataset_type}` first."
        )

    entries: List[DatasetEntry] = []
    model_order: List[str] = []
    parsing_rows = False

    with file_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            if "Patient_ID" in line and "| Actual" in line:
                # Header row lists abbreviated model names (LR, RF, etc.).
                parts = [part.strip() for part in line.split("|")]
                model_order = [
                    MODEL_ABBREVIATIONS.get(part.strip())
                    for part in parts[2:]
                    if part.strip()
                ]
                parsing_rows = True
                continue

            if not parsing_rows:
                continue

            if line.startswith("MODEL ACCURACY SUMMARY"):
                break

            striped = line.strip()
            if not striped or set(striped) <= {"-", "="}:
                continue

            if "|" not in line:
                continue

            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 3:
                continue

            patient_column = parts[0]
            actual_label = parts[1]
            if actual_label not in {"MTC", "No_MTC"}:
                continue

            patient_key = _normalize_patient_key(patient_column)

            prediction_map: Dict[str, str] = {}
            for model_name, prediction in zip(model_order, parts[2:]):
                if not model_name:
                    continue
                prediction_map[model_name] = prediction.strip()

            entries.append(
                DatasetEntry(
                    patient_key=patient_key,
                    actual_label=actual_label,
                    predictions=prediction_map,
                )
            )

    return entries


def extract_positive_outcomes(
    entries: Iterable[DatasetEntry], model_key: str
) -> List[int]:
    """
    Return binary recall outcomes (1 = correctly predicted MTC) for the given model.
    """

    outcomes: List[int] = []
    for entry in entries:
        if entry.actual_label != "MTC":
            continue
        status = entry.predictions.get(model_key)
        if status in {"OK", "TP"}:
            outcomes.append(1)
        elif status in {"XX", "FP", "FN"}:
            outcomes.append(0)
    return outcomes


def build_positive_maps(
    entries: Iterable[DatasetEntry], model_key: str
) -> Dict[str, Deque[int]]:
    """
    Build a dictionary of patient keys to recall outcomes so that McNemar's test can
    be computed on overlapping patients (if any).
    """

    mapping: Dict[str, Deque[int]] = defaultdict(deque)
    for entry in entries:
        if entry.actual_label != "MTC":
            continue
        status = entry.predictions.get(model_key)
        if status is None:
            continue
        mapping[entry.patient_key].append(1 if status in {"OK", "TP"} else 0)
    return mapping


def _mcnemar_p_value(b: int, c: int, exact: bool) -> float | None:
    """Return McNemar p-value using binomial or chi-squared approximation."""

    discordant = b + c
    if discordant == 0:
        return None

    if exact or discordant < 25:
        # two-sided binomial test on the smaller count
        result = stats.binomtest(min(b, c), discordant, 0.5, alternative="two-sided")
        return float(result.pvalue)

    # Chi-squared with continuity correction
    statistic = (abs(b - c) - 1) ** 2 / discordant
    p_value = stats.chi2.sf(statistic, 1)
    return float(p_value)


def run_mcnemar_test(
    original_entries: Iterable[DatasetEntry],
    expanded_entries: Iterable[DatasetEntry],
    model_key: str,
) -> Dict[str, float | int | None]:
    """
    Run McNemar's test on overlapping positive patients, if they exist.

    Returns a dictionary with the computed table, p-value, and number of paired cases.
    """

    orig_map = build_positive_maps(original_entries, model_key)
    exp_map = build_positive_maps(expanded_entries, model_key)

    common_keys = sorted(set(orig_map) & set(exp_map))
    both_correct = both_wrong = orig_only = exp_only = 0
    pair_count = 0

    for key in common_keys:
        left = orig_map[key]
        right = exp_map[key]
        while left and right:
            pair_count += 1
            orig_status = left.popleft()
            exp_status = right.popleft()
            if orig_status == 1 and exp_status == 1:
                both_correct += 1
            elif orig_status == 1 and exp_status == 0:
                orig_only += 1
            elif orig_status == 0 and exp_status == 1:
                exp_only += 1
            else:
                both_wrong += 1

    if pair_count == 0:
        return {
            "pairs": 0,
            "table": ((0, 0), (0, 0)),
            "p_value": None,
            "effect_size": None,
            "note": "No overlapping positive patients; McNemar's test not applicable.",
        }

    contingency = np.array([[both_correct, orig_only], [exp_only, both_wrong]])
    discordant = orig_only + exp_only

    # Use the exact binomial variant when discordant pairs are scarce.
    exact = discordant < 25
    p_value = _mcnemar_p_value(orig_only, exp_only, exact)
    odds_effect = None
    if discordant > 0:
        odds_effect = (orig_only - exp_only) / discordant

    return {
        "pairs": pair_count,
        "table": tuple(tuple(row) for row in contingency.tolist()),
        "p_value": p_value,
        "effect_size": odds_effect,
        "note": None,
    }


def permutation_test(
    original_outcomes: List[int],
    expanded_outcomes: List[int],
    n_permutations: int,
    random_state: int = 42,
) -> Dict[str, float | None]:
    """Two-sided permutation test on the difference in recall."""

    x = np.array(original_outcomes, dtype=float)
    y = np.array(expanded_outcomes, dtype=float)

    if len(x) == 0 or len(y) == 0:
        return {
            "p_value": None,
            "observed_difference": None,
        }

    observed_diff = float(x.mean() - y.mean())
    combined = np.concatenate([x, y])
    n_x = len(x)

    rng = np.random.default_rng(random_state)
    extreme = 0

    for i in range(n_permutations):
        shuffled = rng.permutation(combined)
        perm_x = shuffled[:n_x]
        perm_y = shuffled[n_x:]
        diff = perm_x.mean() - perm_y.mean()
        if abs(diff) >= abs(observed_diff):
            extreme += 1

    p_value = (extreme + 1) / (n_permutations + 1)

    return {
        "p_value": p_value,
        "observed_difference": observed_diff,
    }


def bootstrap_ci(
    original_outcomes: List[int],
    expanded_outcomes: List[int],
    n_iterations: int,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap the recall drop (original - expanded) to obtain a 95% CI.
    """

    x = np.array(original_outcomes, dtype=float)
    y = np.array(expanded_outcomes, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return {"mean": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}

    rng = np.random.default_rng(random_state)
    diffs = np.empty(n_iterations, dtype=float)

    for i in range(n_iterations):
        sample_x = rng.choice(x, size=len(x), replace=True)
        sample_y = rng.choice(y, size=len(y), replace=True)
        diffs[i] = sample_x.mean() - sample_y.mean()

    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean": float(diffs.mean()),
        "std": float(diffs.std(ddof=1)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def cohens_h(p1: float, p2: float) -> float:
    """Effect size (difference between proportions) using Cohen's h."""

    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def summarize_significance(p_value: float | None) -> Dict[float, str]:
    """Return textual yes/no for each alpha level."""

    summary: Dict[float, str] = {}
    for alpha in ALPHAS:
        if p_value is None:
            summary[alpha] = "N/A"
        else:
            summary[alpha] = "Yes" if p_value < alpha else "No"
    return summary


def format_percentage(value: float) -> str:
    """Helper to display percentages consistently."""

    if np.isnan(value):
        return "N/A"
    return f"{value * 100:.1f}%"


def run_statistical_tests(
    permutations: int,
    bootstrap_iterations: int,
) -> Dict[str, Dict[str, object]]:
    """Compute all requested statistics for every model."""

    original_entries = parse_prediction_table("original")
    expanded_entries = parse_prediction_table("expanded")

    results: Dict[str, Dict[str, object]] = {}
    for model_key in MODEL_NAMES:
        orig_outcomes = extract_positive_outcomes(original_entries, model_key)
        exp_outcomes = extract_positive_outcomes(expanded_entries, model_key)

        recall_orig = float(np.mean(orig_outcomes)) if orig_outcomes else float("nan")
        recall_exp = float(np.mean(exp_outcomes)) if exp_outcomes else float("nan")
        recall_drop = recall_orig - recall_exp

        perm = permutation_test(orig_outcomes, exp_outcomes, permutations)
        boot = bootstrap_ci(orig_outcomes, exp_outcomes, bootstrap_iterations)
        mcnemar = run_mcnemar_test(original_entries, expanded_entries, model_key)

        effect = (
            cohens_h(recall_orig, recall_exp)
            if np.isfinite(recall_drop)
            else float("nan")
        )

        results[model_key] = {
            "recall_original": recall_orig,
            "recall_expanded": recall_exp,
            "n_original": len(orig_outcomes),
            "n_expanded": len(exp_outcomes),
            "recall_drop": recall_drop,
            "permutation": perm,
            "bootstrap": boot,
            "mcnemar": mcnemar,
            "effect_size": effect,
            "significance": summarize_significance(perm["p_value"]),
        }

    return results


def write_report(results: Dict[str, Dict[str, object]], output_path: Path) -> None:
    """Persist the statistical findings to disk."""

    lines: List[str] = []
    lines.append("STATISTICAL SIGNIFICANCE TESTS: ORIGINAL VS EXPANDED DATASETS")
    lines.append("=" * 61)
    lines.append("")

    significant_models: List[str] = []

    for model_key in MODEL_NAMES:
        model_result = results[model_key]
        perm = model_result["permutation"]
        boot = model_result["bootstrap"]
        mcnemar = model_result["mcnemar"]
        significance = model_result["significance"]

        lines.append(f"Model: {MODEL_NAMES[model_key]}")
        lines.append(
            f"- Original Recall: {format_percentage(model_result['recall_original'])} "
            f"(n={model_result['n_original']})"
        )
        lines.append(
            f"- Expanded Recall: {format_percentage(model_result['recall_expanded'])} "
            f"(n={model_result['n_expanded']})"
        )
        lines.append(
            f"- Recall Drop (original - expanded): "
            f"{format_percentage(model_result['recall_drop'])}"
        )
        lines.append(
            f"- Bootstrap Mean Drop: {format_percentage(boot['mean'])} "
            f"(Std: {boot['std']:.3f}, 95% CI: "
            f"[{format_percentage(boot['ci_lower'])}, {format_percentage(boot['ci_upper'])}])"
        )

        if mcnemar["p_value"] is None:
            lines.append(f"- McNemar's Test: {mcnemar['note']}")
        else:
            lines.append(
                f"- McNemar's Test p-value: {mcnemar['p_value']:.4g} (pairs={mcnemar['pairs']}, "
                f"effect={mcnemar['effect_size']:.3f})"
            )

        p_value = perm["p_value"]
        if p_value is None:
            lines.append(
                "- Permutation Test: Not enough positive cases for comparison."
            )
        else:
            lines.append(f"- Permutation Test p-value: {p_value:.4g}")
            lines.append(
                "- Significance (alpha=0.05/0.01/0.001): "
                + "/".join(f"{alpha:.3f}:{significance[alpha]}" for alpha in ALPHAS)
            )
            if p_value < 0.05:
                significant_models.append(MODEL_NAMES[model_key])

        effect = model_result["effect_size"]
        effect_text = f"{effect:.3f}" if np.isfinite(effect) else "N/A"
        lines.append(f"- Effect Size (Cohen's h): {effect_text}")
        lines.append("")

    if significant_models:
        conclusion = (
            "Permutation testing indicates statistically significant recall drops for: "
            + ", ".join(significant_models)
            + "."
        )
    else:
        conclusion = "Permutation testing did not flag any recall drops as statistically significant."

    lines.append("Overall Conclusion:")
    lines.append(conclusion)
    lines.append("")
    lines.append(
        "McNemar's test could not be applied because the original and expanded test sets "
        "share no overlapping positive patients; therefore, we rely on permutation and "
        "bootstrap analyses to quantify uncertainty."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate statistical significance of recall differences."
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="Number of shuffles for the permutation test (default: 10000).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for the recall drop CI (default: 1000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/statistical_significance_tests.txt"),
        help="Output path for the textual summary.",
    )

    args = parser.parse_args()

    results = run_statistical_tests(
        permutations=args.permutations,
        bootstrap_iterations=args.bootstrap,
    )
    write_report(results, args.output)

    # Print concise summary to stdout for quick inspection.
    for model_key, stats_dict in results.items():
        perm = stats_dict["permutation"]
        drop_pct = (
            stats_dict["recall_drop"] * 100
            if np.isfinite(stats_dict["recall_drop"])
            else float("nan")
        )
        p_value = perm["p_value"]
        p_text = f"{p_value:.4g}" if p_value is not None else "N/A"
        print(
            f"{MODEL_NAMES[model_key]}: recall drop={drop_pct:.1f} pp, "
            f"permutation p-value={p_text}"
        )


if __name__ == "__main__":
    main()