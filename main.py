import subprocess
import sys
import os
import argparse

def run_module(module_name, description, args=None):
    """run a python module and handle errors"""
    print(f"\n{'='*60}")
    print(f"EXECUTING: {description}")
    print(f"{'='*60}")
    
    try:
        # build command with arguments
        cmd = [sys.executable, module_name]
        if args:
            cmd.extend(args)
            
        result = subprocess.run(cmd, 
                              capture_output=True, text=True, cwd=os.getcwd())
        
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

def main(model_type='logistic'):
    """main orchestration function"""
    print("=" * 80)
    print("RET K666N MUTATION - MEN2 SYNDROME PREDICTION PIPELINE")
    print("=" * 80)
    print("Starting comprehensive ML pipeline for genetic disease prediction...")
    
    # determine model description
    if model_type == 'random_forest':
        model_desc = "Random Forest"
    else:
        model_desc = "Logistic Regression"
    
    print(f"Selected model: {model_desc}")
    
    # define pipeline steps
    pipeline_steps = [
        ("src/create_datasets.py", "Dataset Creation - Extract and structure research data", None),
        ("src/data_analysis.py", "Data Analysis - Generate statistics and visualizations", None),
        ("src/data_expansion.py", "Data Expansion - Create synthetic controls and expand dataset", None),
        ("src/train_model.py", f"Model Training - Train {model_desc} with cross-validation", [f"--m={model_type}"]),
        ("src/test_model.py", f"Model Testing - Evaluate {model_desc} performance on test set", [f"--m={model_type}"])
    ]
    
    # execute each step
    for module_name, description, args in pipeline_steps:
        success = run_module(module_name, description, args)
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
    print("- data/ret_k666n_training_data.csv")
    print("- data/ret_k666n_expanded_training_data.csv") 
    print("- data/men2_case_control_dataset.csv")
    if model_type == 'random_forest':
        print("- data/random_forest_model.pkl")
    else:
        print("- data/logistic_model.pkl")
    print()
    print("The trained model can now be used for RET K666N mutation")
    print("and MEN2 syndrome risk prediction in new patients.")
    
    return True

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description='run complete mtc prediction pipeline')
    parser.add_argument('--m', '--model', type=str, default='l', 
                       choices=['l', 'r', 'logistic', 'random_forest'],
                       help='model type: l/logistic for logistic regression (default), r/random_forest for random forest')
    
    args = parser.parse_args()
    
    # determine model type
    if args.m in ['r', 'random_forest']:
        model_type = 'random_forest'
    else:
        model_type = 'logistic'
    
    success = main(model_type)
    sys.exit(0 if success else 1)