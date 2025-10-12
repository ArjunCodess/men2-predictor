import subprocess
import sys
import os

def run_module(module_name, description):
    """run a python module and handle errors"""
    print(f"\n{'='*60}")
    print(f"EXECUTING: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, module_name], 
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

def main():
    """main orchestration function"""
    print("=" * 80)
    print("RET K666N MUTATION - MEN2 SYNDROME PREDICTION PIPELINE")
    print("=" * 80)
    print("Starting comprehensive ML pipeline for genetic disease prediction...")
    
    # define pipeline steps
    pipeline_steps = [
        ("create_datasets.py", "Dataset Creation - Extract and structure research data"),
        ("data_analysis.py", "Data Analysis - Generate statistics and visualizations"),
        ("data_expansion.py", "Data Expansion - Create synthetic controls and expand dataset"),
        ("train_model.py", "Model Training - Train Logistic Regression with cross-validation"),
        ("test_model.py", "Model Testing - Evaluate performance on test set")
    ]
    
    # execute each step
    for module_name, description in pipeline_steps:
        success = run_module(module_name, description)
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
    print("- model.pkl")
    print()
    print("The trained model can now be used for RET K666N mutation")
    print("and MEN2 syndrome risk prediction in new patients.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)