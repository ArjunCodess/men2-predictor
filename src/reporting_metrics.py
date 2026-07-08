"""Reproducible reporting metrics for the manuscript.

Given confusion-matrix counts (and optional probability scores), this module
produces the exact numbers reported in the manuscript Results and Table 5:
accuracy, sensitivity (recall), specificity, ROC-AUC, and Wilson 95% confidence
intervals for the binomial proportions. Keeping this in one place makes every
reported value auditable and answers reviewer requests to "show where" each
metric and interval comes from.
"""

import math
from sklearn.metrics import roc_auc_score


def wilson_interval(successes, total, z=1.96):
    """Wilson score 95% confidence interval for a binomial proportion.

    Returns (point, lower, upper) as proportions in [0, 1]. For total == 0 the
    point estimate is undefined and returned as None with a [0, 1] interval.
    """
    if total == 0:
        return None, 0.0, 1.0
    phat = successes / total
    denom = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total)) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return phat, lower, upper


def _fmt_pct(value):
    return f"{value * 100:.1f}%"


def metrics_from_counts(tn, fp, fn, tp, y_true=None, y_score=None):
    """Compute reporting metrics from confusion-matrix counts.

    If ``y_true`` and ``y_score`` are provided, ROC-AUC is computed from the
    probability scores (ROC-AUC cannot be reconstructed from counts alone).
    """
    total = tn + fp + fn + tp
    pos = tp + fn
    neg = tn + fp

    acc_p, acc_l, acc_u = wilson_interval(tp + tn, total)
    sens_p, sens_l, sens_u = wilson_interval(tp, pos)
    spec_p, spec_l, spec_u = wilson_interval(tn, neg)

    roc_auc = None
    if y_true is not None and y_score is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            roc_auc = None

    return {
        "counts": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy": {"point": acc_p, "lower": acc_l, "upper": acc_u,
                     "num": tp + tn, "den": total},
        "sensitivity": {"point": sens_p, "lower": sens_l, "upper": sens_u,
                        "num": tp, "den": pos},
        "specificity": {"point": spec_p, "lower": spec_l, "upper": spec_u,
                        "num": tn, "den": neg},
        "roc_auc": roc_auc,
    }


def format_metric(metric):
    """Format a metric dict as 'XX.X% (num/den; low%-high%)'."""
    if metric["point"] is None:
        return "n/a"
    return (f"{_fmt_pct(metric['point'])} ({metric['num']}/{metric['den']}; "
            f"{_fmt_pct(metric['lower'])}-{_fmt_pct(metric['upper'])})")


def format_row(name, model, result):
    """Return a human-readable one-line summary for an analysis row."""
    c = result["counts"]
    cm = f"TN={c['tn']}, FP={c['fp']}, FN={c['fn']}, TP={c['tp']}"
    roc = f"{result['roc_auc']:.3f}" if result["roc_auc"] is not None else "n/a"
    return (
        f"{name} ({model})\n"
        f"  Confusion matrix: {cm}\n"
        f"  Sensitivity: {format_metric(result['sensitivity'])}\n"
        f"  Specificity: {format_metric(result['specificity'])}\n"
        f"  Accuracy:    {format_metric(result['accuracy'])}\n"
        f"  ROC-AUC:     {roc}\n"
    )
