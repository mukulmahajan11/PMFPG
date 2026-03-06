#!/usr/bin/env python3
import subprocess, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)
def main():
    run([sys.executable, "src/generate_synthetic_data.py", "--out_dir", "data/raw"])
    run([sys.executable, "src/task1_data_ingestion_cleaning.py"])
    run([sys.executable, "src/task2_sensor_preprocessing_feature_engineering.py"])
    run([sys.executable, "src/task3_outage_log_analysis_label_generation.py"])
    run([sys.executable, "src/task4_ml_model_selection_initial_training.py"])
    run([sys.executable, "src/task5_model_optimization_validation.py"])
    run([sys.executable, "src/task6_market_mood_index_definition.py"])
if __name__ == "__main__":
    main()
