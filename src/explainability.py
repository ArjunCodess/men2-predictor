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

try:
    import shap
except Exception:
    shap = None

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


def _create_shap_explainer(model, model_type, background_data):
    """create a SHAP explainer tailored to the model type"""
    if shap is None:
        print("SHAP is not installed. Install from requirements.txt to enable explainability.")
        return None

    try:
        if model_type in ['random_forest', 'lightgbm']:
            return shap.TreeExplainer(model.model, data=background_data, model_output="probability")
        if model_type == 'xgboost':
            try:
                return shap.TreeExplainer(model.model, data=background_data, model_output="probability")
            except Exception:
                booster = getattr(model.model, "get_booster", lambda: None)()
                if booster is not None:
                    return shap.TreeExplainer(booster, data=background_data, model_output="probability")
                raise
        if model_type in ['logistic', 'svm']:
            return shap.LinearExplainer(model.model, background_data, feature_perturbation="interventional", model_output="probability")
        return shap.Explainer(model.model, background_data)
    except Exception as e:
        print(f"Unable to initialize SHAP explainer for {model_type}: {e}")
        try:
            print("Falling back to shap.Explainer with predict_proba.")
            return shap.Explainer(getattr(model.model, "predict_proba"), background_data)
        except Exception as fallback_error:
            print(f"Fallback SHAP explainer also failed: {fallback_error}")
            return None


def _extract_shap_array(shap_values, feature_names):
    """normalize shap output to a 2d array of shape (n_samples, n_features)"""
    shap_array = shap_values.values
    if shap_array.ndim == 3:
        if shap_array.shape[1] == len(feature_names):
            class_axis = 2
            class_index = min(1, shap_array.shape[class_axis] - 1)
            shap_array = shap_array[:, :, class_index]
        elif shap_array.shape[2] == len(feature_names):
            class_axis = 1
            class_index = min(1, shap_array.shape[class_axis] - 1)
            shap_array = shap_array[:, class_index, :]
        else:
            shap_array = shap_array.reshape(shap_array.shape[0], -1)
    return shap_array


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
    """create lime plots for a list of explanations"""
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


def generate_shap_explanations(model, X_test, model_type='logistic', dataset_type='expanded', save_beeswarm=False):
    """generate SHAP values, save summaries to results, and plots to charts"""
    print("\n" + "=" * 60)
    print("GENERATING SHAP EXPLANATIONS")
    print("=" * 60)

    if shap is None:
        print("Skipping SHAP generation because the shap package is not available.")
        return None

    if X_test is None or len(X_test) == 0:
        print("No test data available for SHAP computation.")
        return None

    rng = np.random.default_rng(42)

    feature_names = model.feature_columns if model.feature_columns is not None else [f"feature_{i}" for i in range(X_test.shape[1])]
    background_size = min(50, X_test.shape[0])
    sample_size = min(200, X_test.shape[0])

    background_idx = rng.choice(X_test.shape[0], size=background_size, replace=False)
    sample_idx = rng.choice(X_test.shape[0], size=sample_size, replace=False)

    background_data = X_test[background_idx]
    sample_data = X_test[sample_idx]

    explainer = _create_shap_explainer(model, model_type, background_data)
    if explainer is None:
        print("SHAP explainer could not be created; skipping SHAP outputs.")
        return None

    try:
        shap_values = explainer(sample_data)
    except Exception as e:
        print(f"Failed to compute SHAP values: {e}")
        return None

    shap_array = _extract_shap_array(shap_values, feature_names)
    mean_abs_importance = np.abs(shap_array).mean(axis=0)
    top_indices = np.argsort(mean_abs_importance)[::-1]
    top_display = top_indices[:min(10, len(top_indices))]

    root_results = Path("results")
    root_charts = Path("charts")
    dataset_label = "original" if dataset_type == 'original' else "expanded"

    shap_results_dir = root_results / "shap" / model_type
    shap_charts_dir = root_charts / "shap" / model_type
    shap_results_dir.mkdir(parents=True, exist_ok=True)
    shap_charts_dir.mkdir(parents=True, exist_ok=True)

    text_path = shap_results_dir / f"{model_type}_{dataset_label}.txt"
    bar_path = shap_charts_dir / f"{dataset_label}_bar.png"

    with open(text_path, "w") as f:
        f.write("SHAP SUMMARY\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Dataset: {dataset_label}\n")
        f.write(f"Background size: {background_size}\n")
        f.write(f"Sample size: {sample_size}\n")
        f.write("=" * 60 + "\n")
        f.write("Top features by mean |SHAP|:\n")
        for rank, idx in enumerate(top_display, 1):
            f.write(f"{rank:2d}. {feature_names[idx]}: {mean_abs_importance[idx]:.6f}\n")
        f.write("=" * 60 + "\n")

    print(f"Top SHAP drivers ({model_type}, {dataset_label}):")
    for rank, idx in enumerate(top_display, 1):
        print(f"  {rank:2d}. {feature_names[idx]} -> mean |SHAP| = {mean_abs_importance[idx]:.6f}")
    print(f"Saved SHAP summary to {text_path}")

    plt.figure(figsize=(10, 6))
    top_features = [feature_names[idx] for idx in top_display]
    top_values = [mean_abs_importance[idx] for idx in top_display]
    plt.barh(top_features[::-1], top_values[::-1], color="#1f77b4")
    plt.xlabel("Mean |SHAP|")
    plt.title(f"Global SHAP Importance - {model_type} ({dataset_label})")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP bar chart saved to: {bar_path}")

    if save_beeswarm:
        beeswarm_path = shap_charts_dir / f"{dataset_label}_beeswarm.png"
        try:
            shap.summary_plot(shap_values, sample_data, feature_names=feature_names, show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"SHAP beeswarm plot saved to: {beeswarm_path}")
        except Exception as e:
            print(f"Failed to create SHAP beeswarm plot: {e}")

    return shap_values


def compare_lime_shap(lime_exp, shap_values, output_dir=None, save_plot=False, save_summary=False):
    """compare a single lime explanation to shap values"""
    if lime_exp is None or shap_values is None:
        return None

    feature_names = lime_exp.get('feature_names', [])
    if not feature_names:
        return None

    output_path = None
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

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

    sample_idx = lime_exp.get('sample_idx', 'unknown')
    compare_path = None
    if save_plot and output_path is not None:
        plt.figure(figsize=(12, 8))
        plt.barh(y_pos - height / 2, np.abs(lime_vals), height, label="LIME", color="#2ca02c")
        plt.barh(y_pos + height / 2, np.abs(shap_vals), height, label="SHAP", color="#1f77b4")
        plt.yticks(y_pos, labels)
        plt.xlabel("Absolute Feature Impact")
        plt.title("LIME vs SHAP Feature Importance (Local)")
        plt.legend()
        plt.tight_layout()
        compare_path = output_path / f"lime_shap_comparison_{sample_idx}.png"
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

    summary_path = None
    if save_summary and output_path is not None:
        summary_path = output_path / f"lime_shap_consistency_{sample_idx}.txt"
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
        'comparison_plot': str(compare_path) if compare_path is not None else None,
        'summary_path': str(summary_path) if summary_path is not None else None
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


def _get_patient_id(patient, fallback_id):
    """get a readable patient id for outputs"""
    if patient is None:
        return fallback_id
    if pd.notna(patient.get('study_id')):
        return str(patient['study_id'])
    return str(patient.get('source_id', fallback_id))


def _format_counterfactual_change(feature, original_value, counterfactual_value):
    """format counterfactual changes in a clinical style"""
    if feature == 'calcitonin_level_numeric':
        delta = original_value - counterfactual_value
        return f"reduce calcitonin by {abs(delta):.2f} pg/mL (from {original_value:.2f} to {counterfactual_value:.2f})"
    if feature == 'calcitonin_elevated':
        return f"bring calcitonin into normal range (calcitonin_elevated: {int(original_value)} -> {int(counterfactual_value)})"
    if feature == 'thyroid_nodules_present':
        return f"resolve thyroid nodules (nodules_present: {int(original_value)} -> {int(counterfactual_value)})"
    if feature == 'multiple_nodules':
        return f"reduce multiple nodules (multiple_nodules: {int(original_value)} -> {int(counterfactual_value)})"
    return f"adjust {feature}: {original_value} -> {counterfactual_value}"


def _is_actionable_change(feature, original_value, counterfactual_value):
    """ensure counterfactuals align with clinically actionable directions"""
    if feature == 'calcitonin_level_numeric':
        return counterfactual_value < original_value
    if feature == 'calcitonin_elevated':
        return int(original_value) == 1 and int(counterfactual_value) == 0
    if feature == 'thyroid_nodules_present':
        return int(original_value) == 1 and int(counterfactual_value) == 0
    if feature == 'multiple_nodules':
        return int(original_value) == 1 and int(counterfactual_value) == 0
    return True


def _predict_labels(model, X_test_scaled):
    """predict labels using model threshold"""
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    threshold = getattr(model, "threshold", 0.5)
    y_pred = (y_pred_proba >= threshold).astype(int)
    return y_pred, y_pred_proba


def run_lime_explainability(model, X_test_scaled, y_test, test_patients, model_type='logistic', dataset_type='expanded'):
    """generate lime explanations, compare with shap, and counterfactuals"""
    print("\n" + "=" * 60)
    print("GENERATING LIME EXPLAINABILITY")
    print("=" * 60)

    if X_test_scaled is None or len(X_test_scaled) == 0:
        print("No test data available for LIME explainability.")
        return

    feature_names = model.feature_columns if model.feature_columns is not None else [f"feature_{i}" for i in range(X_test_scaled.shape[1])]
    dataset_label = "original" if dataset_type == 'original' else "expanded"

    lime_results_dir = Path("results") / "lime" / model_type
    lime_charts_dir = Path("charts") / "lime" / model_type
    lime_results_dir.mkdir(parents=True, exist_ok=True)
    lime_charts_dir.mkdir(parents=True, exist_ok=True)

    if model.scaler is not None:
        X_test_unscaled = model.scaler.inverse_transform(X_test_scaled)
    else:
        X_test_unscaled = X_test_scaled

    y_pred, y_pred_proba = _predict_labels(model, X_test_scaled)
    y_test_values = y_test.values if hasattr(y_test, "values") else np.array(y_test)

    correct_indices = np.where(y_pred == y_test_values)[0]
    false_negatives = np.where((y_test_values == 1) & (y_pred == 0))[0]
    false_positives = np.where((y_test_values == 0) & (y_pred == 1))[0]

    correct_samples = list(correct_indices[:5])
    misclassified_samples = list(false_negatives[:5])
    if len(misclassified_samples) < 5:
        remaining = 5 - len(misclassified_samples)
        misclassified_samples += list(false_positives[:remaining])

    lime_explanations = []
    selected_samples = correct_samples + misclassified_samples

    if not selected_samples:
        print("No samples available for LIME explanation.")
        return

    print(f"Selected {len(correct_samples)} correct and {len(misclassified_samples)} misclassified samples for LIME.")

    for sample_idx in selected_samples:
        lime_exp = generate_lime_explanation(model, X_test_unscaled, y_test, sample_idx, feature_names)
        if lime_exp is None:
            continue

        patient = test_patients.iloc[sample_idx] if test_patients is not None else None
        lime_exp['patient_id'] = _get_patient_id(patient, f"P_{sample_idx+1}")
        lime_explanations.append(lime_exp)

        status = "correct" if lime_exp['is_correct'] else "misclassified"
        print(f"\nLIME sample {sample_idx} ({status}) - predicted risk: {lime_exp['pred_proba']:.3f}")
        top_features = lime_exp['lime_list'][:3]
        for rank, (feature_desc, weight) in enumerate(top_features, 1):
            print(f"  {rank}. {feature_desc} ({weight:.4f})")

    if not lime_explanations:
        print("LIME explanations could not be generated.")
        return

    create_lime_visualizations(lime_explanations, lime_charts_dir)

    shap_comparisons = []
    shap_values_by_idx = {}
    if shap is None:
        print("SHAP is not installed. Skipping LIME vs SHAP comparison.")
    else:
        rng = np.random.default_rng(42)
        background_size = min(50, X_test_scaled.shape[0])
        background_idx = rng.choice(X_test_scaled.shape[0], size=background_size, replace=False)
        explainer = _create_shap_explainer(model, model_type, X_test_scaled[background_idx])

        if explainer is not None:
            for exp_data in lime_explanations:
                sample_idx = exp_data['sample_idx']
                try:
                    shap_value = explainer(X_test_scaled[[sample_idx]])
                    shap_array = _extract_shap_array(shap_value, feature_names)
                    shap_vector = shap_array[0]
                    exp_data['shap_values'] = shap_vector
                    shap_values_by_idx[sample_idx] = shap_vector

                    comparison = compare_lime_shap(exp_data, shap_vector)
                    if comparison is not None:
                        shap_comparisons.append(comparison)
                except Exception as e:
                    print(f"Failed to compute SHAP for sample {sample_idx}: {e}")

    if shap_values_by_idx:
        name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        lime_matrix = np.zeros((len(lime_explanations), len(feature_names)))
        shap_matrix = np.zeros((len(lime_explanations), len(feature_names)))

        for i, exp_data in enumerate(lime_explanations):
            for feature, weight in exp_data.get('lime_weights', {}).items():
                idx = name_to_idx.get(feature)
                if idx is not None:
                    lime_matrix[i, idx] = weight
            shap_matrix[i, :] = shap_values_by_idx.get(exp_data['sample_idx'], np.zeros(len(feature_names)))

        lime_mean = np.abs(lime_matrix).mean(axis=0)
        shap_mean = np.abs(shap_matrix).mean(axis=0)
        combined_rank = np.argsort(lime_mean + shap_mean)[::-1][:min(10, len(feature_names))]

        plt.figure(figsize=(12, 8))
        labels = [feature_names[idx] for idx in combined_rank]
        y_pos = np.arange(len(labels))
        bar_height = 0.35
        plt.barh(y_pos - bar_height / 2, lime_mean[combined_rank], bar_height, label="LIME", color="#2ca02c")
        plt.barh(y_pos + bar_height / 2, shap_mean[combined_rank], bar_height, label="SHAP", color="#1f77b4")
        plt.yticks(y_pos, labels)
        plt.xlabel("Mean Absolute Feature Impact")
        plt.title(f"LIME vs SHAP - Global Importance ({model_type}, {dataset_label})")
        plt.legend()
        plt.tight_layout()
        global_compare_path = lime_charts_dir / f"lime_shap_global_{dataset_label}.png"
        plt.savefig(global_compare_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"LIME vs SHAP global comparison saved to: {global_compare_path}")

    consistency_path = lime_results_dir / f"{model_type}_{dataset_label}_consistency.txt"
    if shap_comparisons:
        agree_count = sum(1 for c in shap_comparisons if c['agreement_level'] == 'agree')
        partial_count = sum(1 for c in shap_comparisons if c['agreement_level'] == 'partial')
        disagree_count = sum(1 for c in shap_comparisons if c['agreement_level'] == 'disagree')

        with open(consistency_path, "w") as f:
            f.write("LIME vs SHAP CONSISTENCY SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model: {model_type}\n")
            f.write(f"Dataset: {dataset_label}\n")
            f.write(f"Samples compared: {len(shap_comparisons)}\n")
            f.write(f"Agree: {agree_count}\n")
            f.write(f"Partial: {partial_count}\n")
            f.write(f"Disagree: {disagree_count}\n")

        print(f"LIME vs SHAP consistency summary saved to: {consistency_path}")

    actionable_features = [f for f in ['calcitonin_level_numeric', 'calcitonin_elevated', 'thyroid_nodules_present', 'multiple_nodules'] if f in feature_names]
    counterfactual_lines = []
    counterfactual_path = lime_results_dir / f"{model_type}_{dataset_label}_counterfactuals.txt"

    if actionable_features:
        high_risk_indices = [idx for idx in selected_samples if y_pred_proba[idx] >= 0.20]

        for sample_idx in high_risk_indices:
            cf_df = generate_counterfactuals(model, X_test_unscaled, y_test, sample_idx, feature_names, features_to_vary=actionable_features)
            if cf_df is None or cf_df.empty:
                continue

            original = pd.Series(X_test_unscaled[sample_idx], index=feature_names)
            counterfactual = cf_df.iloc[0]
            changes = []
            for feature in actionable_features:
                orig_val = original[feature]
                cf_val = counterfactual[feature]
                if np.isnan(orig_val) or np.isnan(cf_val):
                    continue
                if abs(orig_val - cf_val) > 1e-6 and _is_actionable_change(feature, orig_val, cf_val):
                    change_text = _format_counterfactual_change(feature, orig_val, cf_val)
                    changes.append((feature, abs(orig_val - cf_val), change_text))

            changes = sorted(changes, key=lambda x: x[1], reverse=True)[:3]
            if changes:
                patient = test_patients.iloc[sample_idx] if test_patients is not None else None
                patient_id = _get_patient_id(patient, f"P_{sample_idx+1}")
                header = f"Patient {patient_id} (sample {sample_idx}) - predicted risk {y_pred_proba[sample_idx]:.3f}"
                counterfactual_lines.append(header)
                print(f"\nCounterfactuals for {header}:")
                for rank, (_, _, change_text) in enumerate(changes, 1):
                    line = f"  {rank}. To reduce risk, patient would need to {change_text}"
                    counterfactual_lines.append(line)
                    print(line)
                counterfactual_lines.append("")

    if counterfactual_lines:
        with open(counterfactual_path, "w") as f:
            f.write("COUNTERFACTUAL ANALYSIS (HIGH-RISK CASES)\n")
            f.write("=" * 60 + "\n")
            f.write("High-risk definition: predicted probability >= 0.20\n\n")
            for line in counterfactual_lines:
                f.write(line + "\n")

        print(f"Counterfactual analysis saved to: {counterfactual_path}")

    summary_path = Path("results") / "explainability" / "explainability_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "a") as f:
        f.write("EXPLAINABILITY SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Dataset: {dataset_label}\n")
        f.write(f"LIME explained: {len(lime_explanations)} samples\n")
        f.write(f"Correct: {len(correct_samples)}, Misclassified: {len(misclassified_samples)}\n")
        if shap_comparisons:
            f.write(f"LIME vs SHAP agreements: {agree_count}, partial: {partial_count}, disagree: {disagree_count}\n")
        if counterfactual_lines:
            f.write("\nCounterfactual highlights:\n")
            for line in counterfactual_lines:
                f.write(line + "\n")
        f.write("=" * 60 + "\n\n")

    print(f"Explainability summary updated at: {summary_path}")


def run_explainability(model, X_test_scaled, y_test, test_patients, model_type='logistic', dataset_type='expanded'):
    """run SHAP + LIME explainability for a model"""
    generate_shap_explanations(model, X_test_scaled, model_type=model_type, dataset_type=dataset_type, save_beeswarm=False)
    run_lime_explainability(model, X_test_scaled, y_test, test_patients, model_type, dataset_type)
