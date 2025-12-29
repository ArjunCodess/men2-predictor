import subprocess
import sys
import os
import argparse
from pathlib import Path

def run_module(module_name, description, args=None, log_file=None, dataset_type=None):
    """run a python module and handle errors"""
    print(f"\n{'='*60}")
    print(f"EXECUTING: {description}")
    print(f"{'='*60}")

    try:
        # build command with arguments
        cmd = [sys.executable, module_name]
        if args:
            cmd.extend(args)
        # Add dataset type if specified
        if dataset_type:
            cmd.extend([f"--d={dataset_type}"])

        result = subprocess.run(cmd,
                              capture_output=True, text=True, cwd=os.getcwd())

        # Save to log file if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"{'='*60}\n")
                f.write(f"EXECUTION LOG: {description}\n")
                f.write(f"{'='*60}\n\n")
                if result.stdout:
                    f.write(result.stdout)
                if result.stderr:
                    f.write(f"\nSTDERR:\n{result.stderr}")
                f.write(f"\n\nReturn Code: {result.returncode}\n")

            print(f"Log saved to: {log_file}")
        else:
            # Print to console if no log file specified
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"STDERR: {result.stderr}")

        if result.returncode != 0:
            print(f"ERROR: Module {module_name} failed with return code {result.returncode}")
            return False

        return True

    except Exception as e:
        print(f"ERROR: Failed to execute {module_name}: {str(e)}")
        return False

def extract_model_metrics(model_type, dataset_type='expanded'):
    """Extract metrics from test results file"""
    results_file = Path('results') / f'{model_type}_{dataset_type}_test_results.txt'

    if not results_file.exists():
        return None

    metrics = {}
    try:
        with open(results_file, 'r') as f:
            content = f.read()

            # Extract key metrics using simple parsing
            for line in content.split('\n'):
                # Split on colon first, then extract the main value before any parentheses
                if 'Accuracy:' in line:
                    value_str = line.split(':')[1].strip()
                    # Extract the numeric value before any parentheses
                    if '(' in value_str:
                        metrics['accuracy'] = float(value_str.split('(')[0].strip())
                    else:
                        metrics['accuracy'] = float(value_str)
                elif line.strip().startswith('Precision:'):
                    value_str = line.split(':')[1].strip()
                    if '(' in value_str:
                        metrics['precision'] = float(value_str.split('(')[0].strip())
                    else:
                        metrics['precision'] = float(value_str)
                elif 'Average Precision:' in line:
                    value_str = line.split(':')[1].strip()
                    if '(' in value_str:
                        metrics['avg_precision'] = float(value_str.split('(')[0].strip())
                    else:
                        metrics['avg_precision'] = float(value_str)
                elif 'Recall:' in line:
                    value_str = line.split(':')[1].strip()
                    if '(' in value_str:
                        metrics['recall'] = float(value_str.split('(')[0].strip())
                    else:
                        metrics['recall'] = float(value_str)
                elif 'F1 Score:' in line:
                    value_str = line.split(':')[1].strip()
                    if '(' in value_str:
                        metrics['f1_score'] = float(value_str.split('(')[0].strip())
                    else:
                        metrics['f1_score'] = float(value_str)
                elif 'ROC AUC:' in line:
                    value_str = line.split(':')[1].strip()
                    if '(' in value_str:
                        metrics['roc_auc'] = float(value_str.split('(')[0].strip())
                    else:
                        metrics['roc_auc'] = float(value_str)
    except Exception as e:
        print(f"Warning: Could not extract metrics for {model_type}: {e}")
        return None

    # Verify all required metrics were found
    required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    if not all(metric in metrics for metric in required_metrics):
        print(f"Warning: Not all metrics found for {model_type}. Found: {list(metrics.keys())}")
        return None

    return metrics

def print_comparison_table(results):
    """Print a formatted comparison table of all model results"""
    print("\n" + "=" * 120)
    print("MODEL COMPARISON RESULTS")
    print("=" * 120)

    # Define table headers
    headers = ["Model", "Dataset", "Accuracy", "Precision", "Avg Precision", "Recall", "F1 Score", "ROC AUC", "Status"]
    col_widths = [21, 15, 12, 12, 17, 12, 12, 12, 15]

    # Print header
    header_row = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * 120)

    # Print each model's results
    for key, data in results.items():
        # Parse key (format: "model_type_dataset_type")
        if key.endswith('_expanded'):
            model_type = key.replace('_expanded', '')
            dataset_label = "Expanded"
        elif key.endswith('_original'):
            model_type = key.replace('_original', '')
            dataset_label = "Original"
        else:
            model_type = key
            dataset_label = "Expanded"

        model_names = {
            'logistic': 'Logistic Regression',
            'random_forest': 'Random Forest',
            'xgboost': 'XGBoost',
            'lightgbm': 'LightGBM',
            'svm': 'SVM'
        }

        model_name = model_names.get(model_type, model_type)
        status = "SUCCESS" if data['success'] else "FAILED"

        if data['metrics']:
            row = [
                model_name,
                dataset_label,
                f"{data['metrics'].get('accuracy', 0):.4f}",
                f"{data['metrics'].get('precision', 0):.4f}",
                f"{data['metrics'].get('avg_precision', 0):.4f}",
                f"{data['metrics'].get('recall', 0):.4f}",
                f"{data['metrics'].get('f1_score', 0):.4f}",
                f"{data['metrics'].get('roc_auc', 0):.4f}",
                status
            ]
        else:
            row = [model_name, dataset_label, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", status]

        row_str = "".join(f"{str(val):<{w}}" for val, w in zip(row, col_widths))
        print(row_str)

    print("=" * 120)

    # Find best performing model
    best_model = None
    best_f1 = -1
    for key, data in results.items():
        if data['metrics'] and data['success']:
            f1 = data['metrics'].get('f1_score', 0)
            if f1 > best_f1:
                best_f1 = f1
                best_model = key

    if best_model:
        # Parse key for display
        if best_model.endswith('_expanded'):
            model_type = best_model.replace('_expanded', '')
            dataset_label = "Expanded"
        elif best_model.endswith('_original'):
            model_type = best_model.replace('_original', '')
            dataset_label = "Original"
        else:
            model_type = best_model
            dataset_label = "Expanded"

        model_names = {
            'logistic': 'Logistic Regression',
            'random_forest': 'Random Forest',
            'xgboost': 'XGBoost',
            'lightgbm': 'LightGBM',
            'svm': 'SVM'
        }
        print(f"\nBest performing model: {model_names[model_type]} on {dataset_label} (F1 Score: {best_f1:.4f})")
    print()

def run_all_models(dataset_type='expanded', skip_ci=False):
    """Run all model types and compare results"""
    print("=" * 80)
    print("MULTI-VARIANT RET MUTATION - MEN2 SYNDROME PREDICTION PIPELINE")
    print("=" * 80)

    # Determine what datasets to run
    if dataset_type == 'both':
        dataset_types = ['expanded', 'original']
        print("Running ALL MODELS on BOTH DATASETS for comprehensive comparison...")
    else:
        dataset_types = [dataset_type]
        dataset_label = "EXPANDED" if dataset_type == 'expanded' else "ORIGINAL"
        print(f"Running ALL MODELS on {dataset_label} for comprehensive comparison...")
    print()

    model_types = ['logistic', 'random_forest', 'xgboost', 'lightgbm', 'svm']
    results = {}

    # Create results directory for logs
    os.makedirs('results/logs', exist_ok=True)
    summary_path = Path("results") / "explainability_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("")

    # Run data preparation steps once (common to all models)
    print("\n" + "=" * 80)
    print("STEP 1: DATA PREPARATION (Common for all models)")
    print("=" * 80)

    prep_steps = [
        ("src/create_datasets.py", "Dataset Creation - Extract and structure research data", None),
        ("src/data_analysis.py", "Data Analysis - Generate statistics and visualizations", None),
        ("src/data_expansion.py", "Data Expansion - Create synthetic controls and expand dataset", None),
    ]

    for i, (module_name, description, args) in enumerate(prep_steps):
        log_file = f"results/logs/data_preparation_step{i+1}.log"
        success = run_module(module_name, description, args, log_file=log_file)
        if not success:
            print(f"\n{'!'*60}")
            print(f"DATA PREPARATION FAILED AT: {description}")
            print(f"Check log file: {log_file}")
            print(f"{'!'*60}")
            return False

    # Run each model type on each dataset type
    print("\n" + "=" * 80)
    print("STEP 2: TRAINING AND TESTING ALL MODELS")
    print("=" * 80)

    for dt in dataset_types:
        dataset_label = "EXPANDED" if dt == 'expanded' else "ORIGINAL"

        print(f"\n{'='*80}")
        print(f"DATASET TYPE: {dataset_label}")
        print(f"{'='*80}")

        for model_type in model_types:
            model_desc = {
                'logistic': 'Logistic Regression',
                'random_forest': 'Random Forest',
                'xgboost': 'XGBoost',
                'lightgbm': 'LightGBM',
                'svm': 'SVM'
            }[model_type]

            print(f"\n{'-'*80}")
            print(f"Processing: {model_desc} on {dataset_label}")
            print(f"{'-'*80}")

            # Train model - save log to file
            train_log = f"results/logs/{model_type}_{dt}_training.log"
            train_success = run_module(
                "src/train_model.py",
                f"Model Training - Train {model_desc} with cross-validation on {dataset_label}",
                [f"--m={model_type}"],
                log_file=train_log,
                dataset_type=dt
            )

            if not train_success:
                print(f"Training failed! Check log: {train_log}")

            # Test model - save log to file
            test_log = f"results/logs/{model_type}_{dt}_testing.log"
            test_success = run_module(
                "src/test_model.py",
                f"Model Testing - Evaluate {model_desc} performance on test set ({dataset_label})",
                [f"--m={model_type}"],
                log_file=test_log,
                dataset_type=dt
            )

            if not test_success:
                print(f"Testing failed! Check log: {test_log}")

            # Calculate confidence intervals unless skipped
            ci_success = True
            if not skip_ci:
                ci_log = f"results/logs/{model_type}_{dt}_confidence_intervals.log"
                ci_success = run_module(
                    "src/calculate_ci.py",
                    f"Confidence Intervals - Calculate 95% bootstrap CIs for {model_desc} ({dataset_label})",
                    [f"--m={model_type}", f"--iterations=1000"],
                    log_file=ci_log,
                    dataset_type=dt
                )

                if not ci_success:
                    print(f"Confidence interval calculation failed! Check log: {ci_log}")
            else:
                print(f"Skipping confidence interval calculation for {model_desc} (--no-ci flag)")

            # Extract metrics
            metrics = extract_model_metrics(model_type, dt)

            # Store with combined key
            results_key = f"{model_type}_{dt}"
            results[results_key] = {
                'success': train_success and test_success and ci_success,
                'metrics': metrics
            }

            # Print summary for this model
            if train_success and test_success and metrics:
                print(f"\n{model_desc} on {dataset_label} completed successfully!")
                print(f"  - Training log: {train_log}")
                print(f"  - Testing log: {test_log}")
                print(f"  - Metrics: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
                print(f"  - Explainability outputs: results/shap/{model_type}, results/lime/{model_type}, charts/shap/{model_type}, charts/lime/{model_type}")
            else:
                print(f"\n{model_desc} on {dataset_label} encountered issues. Check logs for details.")

    # Print comparison table
    print_comparison_table(results)

    # Final summary
    print("=" * 80)
    print("ALL MODELS PIPELINE COMPLETED!")
    print("=" * 80)
    print("\nModel artifacts saved:")
    print("- data/processed/ret_multivariant_training_data.csv")
    print("- data/processed/ret_multivariant_expanded_training_data.csv")
    print("- data/processed/ret_multivariant_case_control_dataset.csv")
    print("- saved_models/*_{expanded|original}_model.pkl")
    print("\nTest results:")
    print("- results/*_test_results.txt")
    print("\nDetailed logs:")
    print("- results/logs/data_preparation_step1.log")
    print("- results/logs/data_preparation_step2.log")
    print("- results/logs/data_preparation_step3.log")
    print("- results/logs/*_training.log")
    print("- results/logs/*_testing.log")
    print("- results/logs/*_confidence_intervals.log")
    print("\nExplainability outputs:")
    print("- results/shap/<model>/*")
    print("- charts/shap/<model>/*")
    print("- results/lime/<model>/*")
    print("- charts/lime/<model>/*")
    print("- results/explainability_summary.txt")
    print()

    return True

def main(model_type='logistic', dataset_type='expanded', skip_ci=False):
    """main orchestration function"""
    print("=" * 80)
    print("MULTI-VARIANT RET MUTATION - MEN2 SYNDROME PREDICTION PIPELINE")
    print("=" * 80)
    print("Starting comprehensive ML pipeline for genetic disease prediction...")

    # determine model description
    if model_type == 'random_forest':
        model_desc = "Random Forest"
    elif model_type == 'xgboost':
        model_desc = "XGBoost"
    elif model_type == 'lightgbm':
        model_desc = "LightGBM"
    elif model_type == 'svm':
        model_desc = "SVM"
    else:
        model_desc = "Logistic Regression"

    dataset_label = "EXPANDED" if dataset_type == 'expanded' else "ORIGINAL"

    print(f"Selected model: {model_desc}")
    print(f"Dataset type: {dataset_label}")
    if skip_ci:
        print("Skipping confidence interval calculations (--no-ci flag)")

    # define pipeline steps
    pipeline_steps = [
        ("src/create_datasets.py", "Dataset Creation - Extract and structure research data", None),
        ("src/data_analysis.py", "Data Analysis - Generate statistics and visualizations", None),
        ("src/data_expansion.py", "Data Expansion - Create synthetic controls and expand dataset", None),
        ("src/train_model.py", f"Model Training - Train {model_desc} with cross-validation", [f"--m={model_type}"]),
        ("src/test_model.py", f"Model Testing - Evaluate {model_desc} performance on test set", [f"--m={model_type}"])
    ]

    # Add confidence interval step unless skipped
    if not skip_ci:
        pipeline_steps.append(("src/calculate_ci.py", f"Confidence Intervals - Calculate 95% bootstrap CIs for {model_desc}", [f"--m={model_type}"]))

    # execute each step
    for module_name, description, args in pipeline_steps:
        success = run_module(module_name, description, args, dataset_type=dataset_type)
        if not success:
            print(f"\n{'!'*60}")
            print(f"PIPELINE FAILED AT: {description}")
            print(f"{'!'*60}")
            return False

    # pipeline completed successfully
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("All modules executed successfully.")
    print("Model artifacts saved:")
    print("- data/processed/ret_multivariant_training_data.csv")
    print("- data/processed/ret_multivariant_expanded_training_data.csv")
    print("- data/processed/ret_multivariant_case_control_dataset.csv")
    if model_type == 'random_forest':
        print(f"- saved_models/random_forest_{dataset_type}_model.pkl")
    elif model_type == 'xgboost':
        print(f"- saved_models/xgboost_{dataset_type}_model.pkl")
    elif model_type == 'lightgbm':
        print(f"- saved_models/lightgbm_{dataset_type}_model.pkl")
    elif model_type == 'svm':
        print(f"- saved_models/svm_{dataset_type}_model.pkl")
    else:
        print(f"- saved_models/logistic_regression_{dataset_type}_model.pkl")
    print()
    print("The trained model can now be used for multi-variant RET mutation")
    print("and MEN2 syndrome risk prediction in new patients.")
    print()
    print("Explainability outputs:")
    print(f"- results/shap/{model_type}")
    print(f"- charts/shap/{model_type}")
    print(f"- results/lime/{model_type}")
    print(f"- charts/lime/{model_type}")
    print("- results/explainability_summary.txt")

    return True

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description='run complete mtc prediction pipeline')
    parser.add_argument('--m', '--model', type=str, default='l',
                       choices=['l', 'r', 'x', 'g', 's', 'a', 'logistic', 'random_forest', 'xgboost', 'lightgbm', 'svm', 'all'],
                       help='model type: l/logistic (default), r/random_forest, x/xgboost, g/lightgbm, s/svm, a/all (compare all models)')
    parser.add_argument('--d', '--data', type=str, default='e',
                       choices=['e', 'o', 'b', 'expanded', 'original', 'both'],
                       help='dataset type: e/expanded (with controls + SMOTE - default), o/original (paper data only), b/both (run on both datasets)')
    parser.add_argument('--no-ci', action='store_true',
                       help='skip confidence interval calculations for faster execution')

    args = parser.parse_args()

    # Determine dataset type
    if args.d in ['o', 'original']:
        dataset_type = 'original'
    elif args.d in ['b', 'both']:
        dataset_type = 'both'
    else:
        dataset_type = 'expanded'

    # Check if user wants to run all models
    if args.m in ['a', 'all']:
        success = run_all_models(dataset_type, skip_ci=args.no_ci)
    else:
        # determine model type
        if args.m in ['r', 'random_forest']:
            model_type = 'random_forest'
        elif args.m in ['x', 'xgboost']:
            model_type = 'xgboost'
        elif args.m in ['g', 'lightgbm']:
            model_type = 'lightgbm'
        elif args.m in ['s', 'svm']:
            model_type = 'svm'
        else:
            model_type = 'logistic'

        success = main(model_type, dataset_type, skip_ci=args.no_ci)

    stats_success = True
    if success and dataset_type == 'both':
        stats_success = run_module(
            "src/statistical_tests.py",
            "Statistical Significance Tests - Recall comparison for original vs expanded datasets",
            log_file="results/logs/statistical_tests.log"
        )
    elif success:
        print("Skipping statistical significance tests (requires running both datasets).")

    sys.exit(0 if success and stats_success else 1)
