import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

try:
    import dice_ml
except Exception:
    dice_ml = None

_LIME_EXPLAINER_CACHE = {}


def _get_lime_explainer(X_train, feature_names):
    """create or reuse a lime explainer for the given feature set"""
    if LimeTabularExplainer is None:
        return None

    cache_key = (X_train.shape[1], tuple(feature_names))
    if cache_key in _LIME_EXPLAINER_CACHE:
        return _LIME_EXPLAINER_CACHE[cache_key]

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['No MTC', 'MTC'],
        discretize_continuous=True,
        mode='classification',
        random_state=42
    )

    _LIME_EXPLAINER_CACHE[cache_key] = explainer
    return explainer


def _predict_with_scaler(model, X):
    """scale inputs (if needed) and return probabilities"""
    X_array = np.array(X)
    if model.scaler is not None:
        X_array = model.scaler.transform(X_array)
    return model.model.predict_proba(X_array)


def generate_lime_explanation(model, X_test, y_test, sample_idx, feature_names):
    """generate a lime explanation for a single test sample"""
    if LimeTabularExplainer is None:
        print("LIME is not installed. Install from requirements.txt to enable LIME explainability.")
        return None

    if X_test is None or len(X_test) == 0:
        print("No test data available for LIME explanation.")
        return None

    if sample_idx < 0 or sample_idx >= len(X_test):
        print(f"Invalid sample index for LIME: {sample_idx}")
        return None

    X_values = X_test.values if isinstance(X_test, pd.DataFrame) else np.array(X_test)

    explainer = _get_lime_explainer(X_values, feature_names)
    if explainer is None:
        print("Unable to initialize LIME explainer.")
        return None

    predict_fn = lambda x: _predict_with_scaler(model, x)
    try:
        exp = explainer.explain_instance(
            X_values[sample_idx],
            predict_fn,
            num_features=min(10, len(feature_names)),
            labels=(1,)
        )
    except Exception as e:
        print(f"Failed to generate LIME explanation for sample {sample_idx}: {e}")
        return None

    lime_map = exp.as_map()
    lime_weights = {}
    for feature_idx, weight in lime_map.get(1, []):
        if feature_idx < len(feature_names):
            lime_weights[feature_names[feature_idx]] = weight

    y_true = int(y_test.iloc[sample_idx]) if hasattr(y_test, "iloc") else int(y_test[sample_idx])
    proba = predict_fn([X_values[sample_idx]])[0, 1]
    threshold = getattr(model, "threshold", 0.5)
    y_pred = int(proba >= threshold)

    return {
        'sample_idx': sample_idx,
        'true_label': y_true,
        'pred_label': y_pred,
        'pred_proba': float(proba),
        'is_correct': bool(y_true == y_pred),
        'lime_list': exp.as_list(label=1),
        'lime_weights': lime_weights,
        'feature_names': feature_names,
        'explanation': exp
    }


def create_lime_visualizations(explanations, output_dir):
    """create lime plots and html outputs for a list of explanations"""
    if not explanations:
        print("No LIME explanations provided for visualization.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for exp_data in explanations:
        exp = exp_data.get('explanation')
        if exp is None:
            continue

        sample_idx = exp_data.get('sample_idx', 'unknown')
        status = "correct" if exp_data.get('is_correct') else "misclassified"
        pred_proba = exp_data.get('pred_proba', 0.0)

        # png plot
        try:
            fig = exp.as_pyplot_figure(label=1)
            fig.suptitle(f"LIME Explanation - Sample {sample_idx} ({status})\nPredicted risk: {pred_proba:.3f}")
            fig.tight_layout()
            png_path = output_dir / f"lime_sample_{sample_idx}_{status}.png"
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"Failed to save LIME plot for sample {sample_idx}: {e}")

    # global lime importance summary
    feature_names = explanations[0].get('feature_names', [])
    if feature_names:
        name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        lime_matrix = np.zeros((len(explanations), len(feature_names)))

        for i, exp_data in enumerate(explanations):
            for feature, weight in exp_data.get('lime_weights', {}).items():
                idx = name_to_idx.get(feature)
                if idx is not None:
                    lime_matrix[i, idx] = weight

        mean_abs_importance = np.abs(lime_matrix).mean(axis=0)
        top_indices = np.argsort(mean_abs_importance)[::-1][:min(10, len(feature_names))]

        plt.figure(figsize=(10, 6))
        top_features = [feature_names[idx] for idx in top_indices]
        top_values = [mean_abs_importance[idx] for idx in top_indices]
        plt.barh(top_features[::-1], top_values[::-1], color="#2ca02c")
        plt.xlabel("Mean |LIME weight|")
        plt.title("Global LIME Feature Importance (Explained Samples)")
        plt.tight_layout()
        global_path = output_dir / "lime_global_importance.png"
        plt.savefig(global_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"LIME global importance saved to: {global_path}")


def compare_lime_shap(lime_exp, shap_values, output_dir):
    """compare a single lime explanation to shap values"""
    if lime_exp is None or shap_values is None:
        return None

    feature_names = lime_exp.get('feature_names', [])
    if not feature_names:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    lime_vector = np.zeros(len(feature_names))
    for feature, weight in lime_exp.get('lime_weights', {}).items():
        idx = name_to_idx.get(feature)
        if idx is not None:
            lime_vector[idx] = weight

    shap_vector = np.array(shap_values).reshape(-1)
    if len(shap_vector) != len(feature_names):
        return None

    top_k = min(10, len(feature_names))
    lime_top_idx = np.argsort(np.abs(lime_vector))[::-1][:top_k]
    shap_top_idx = np.argsort(np.abs(shap_vector))[::-1][:top_k]
    combined_idx = list(dict.fromkeys(list(lime_top_idx) + list(shap_top_idx)))

    labels = [feature_names[idx] for idx in combined_idx]
    lime_vals = [lime_vector[idx] for idx in combined_idx]
    shap_vals = [shap_vector[idx] for idx in combined_idx]

    y_pos = np.arange(len(labels))
    height = 0.35

    plt.figure(figsize=(12, 8))
    plt.barh(y_pos - height / 2, np.abs(lime_vals), height, label="LIME", color="#2ca02c")
    plt.barh(y_pos + height / 2, np.abs(shap_vals), height, label="SHAP", color="#1f77b4")
    plt.yticks(y_pos, labels)
    plt.xlabel("Absolute Feature Impact")
    plt.title("LIME vs SHAP Feature Importance (Local)")
    plt.legend()
    plt.tight_layout()

    sample_idx = lime_exp.get('sample_idx', 'unknown')
    compare_path = output_dir / f"lime_shap_comparison_{sample_idx}.png"
    plt.savefig(compare_path, dpi=300, bbox_inches="tight")
    plt.close()

    lime_top_set = set(lime_top_idx[:5])
    shap_top_set = set(shap_top_idx[:5])
    overlap = lime_top_set.intersection(shap_top_set)
    sign_matches = 0
    for idx in overlap:
        if np.sign(lime_vector[idx]) == np.sign(shap_vector[idx]):
            sign_matches += 1

    agreement_level = "disagree"
    if len(overlap) >= 3:
        agreement_level = "agree"
    elif len(overlap) >= 1:
        agreement_level = "partial"

    summary_path = output_dir / f"lime_shap_consistency_{sample_idx}.txt"
    with open(summary_path, "w") as f:
        f.write("LIME vs SHAP CONSISTENCY SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Sample: {sample_idx}\n")
        f.write(f"Top-5 overlap: {len(overlap)} features\n")
        f.write(f"Sign agreement on overlap: {sign_matches}/{len(overlap)}\n")
        f.write(f"Agreement level: {agreement_level}\n")

    return {
        'sample_idx': sample_idx,
        'overlap_count': len(overlap),
        'sign_matches': sign_matches,
        'agreement_level': agreement_level,
        'comparison_plot': str(compare_path),
        'summary_path': str(summary_path)
    }


def generate_counterfactuals(model, X_data, y_data, sample_idx, feature_names, features_to_vary=None, total_cfs=1):
    """generate counterfactuals using dice-ml for a single sample"""
    if dice_ml is None:
        print("dice-ml is not installed. Install from requirements.txt to enable counterfactuals.")
        return None

    if X_data is None or len(X_data) == 0:
        print("No data available for counterfactual generation.")
        return None

    df = pd.DataFrame(X_data, columns=feature_names)
    df['mtc_diagnosis'] = y_data.values if hasattr(y_data, "values") else np.array(y_data)

    continuous_features = []
    for col in feature_names:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) > 2:
            continuous_features.append(col)

    class ScaledModelWrapper:
        def __init__(self, model, scaler, feature_names):
            self.model = model
            self.scaler = scaler
            self.feature_names = feature_names

        def predict_proba(self, X):
            if isinstance(X, pd.DataFrame):
                X = X[self.feature_names].values
            if self.scaler is not None:
                X = self.scaler.transform(X)
            return self.model.predict_proba(X)

    wrapped_model = ScaledModelWrapper(model.model, model.scaler, feature_names)
    data = dice_ml.Data(
        dataframe=df,
        continuous_features=continuous_features,
        outcome_name='mtc_diagnosis'
    )
    dice_model = dice_ml.Model(model=wrapped_model, backend="sklearn")

    try:
        dice = dice_ml.Dice(data, dice_model, method="random")
        query_instance = df.iloc[[sample_idx]].drop(columns=['mtc_diagnosis'])
        dice_exp = dice.generate_counterfactuals(
            query_instance,
            total_CFs=total_cfs,
            desired_class="opposite",
            features_to_vary=features_to_vary
        )
        if not dice_exp.cf_examples_list:
            return None
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df
        if cf_df is None or cf_df.empty:
            return None
        return cf_df
    except Exception as e:
        print(f"Failed to generate counterfactuals for sample {sample_idx}: {e}")
        return None
