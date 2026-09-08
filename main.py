"""
CLI Benchmark Runner for Enhanced Fuzzy Information Decomposition (FID) Data Recovery.
"""

import os
import argparse
import numpy as np
import pandas as pd

from src.core.fid_engine import FIDRecovery
from src.data.preprocessor import DataPreprocessor, EnhancedDataPreprocessor
from src.evaluation.metrics import RecoveryEvaluator

def generate_sample_forestfires() -> pd.DataFrame:
    """Generate synthetic forest fires benchmark dataset if local CSV is missing."""
    np.random.seed(42)
    n = 200
    temp = np.random.uniform(5.0, 33.0, n)
    RH = 100 - 2.5 * temp + np.random.normal(0, 5, n)
    wind = np.random.uniform(1.0, 9.0, n)
    rain = np.random.choice([0.0, 0.0, 0.0, 0.2, 0.8], n)
    area = np.exp(temp / 10.0) + np.random.normal(0, 1, n)
    
    df = pd.DataFrame({
        'X': np.random.randint(1, 9, n),
        'Y': np.random.randint(1, 9, n),
        'month': np.random.choice(['mar', 'aug', 'sep', 'jul'], n),
        'day': np.random.choice(['mon', 'sun', 'fri', 'sat'], n),
        'FFMC': np.random.uniform(80.0, 96.0, n),
        'DMC': np.random.uniform(20.0, 290.0, n),
        'DC': np.random.uniform(100.0, 800.0, n),
        'ISI': np.random.uniform(2.0, 20.0, n),
        'temp': temp,
        'RH': RH,
        'wind': wind,
        'rain': rain,
        'area': area
    })
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Fuzzy Information Decomposition (FID) Data Recovery CLI Engine"
    )
    parser.add_argument("--csv", type=str, default="forestfires", help="Dataset name ('forestfires') or path to custom CSV file")
    parser.add_argument("--feature", type=str, default="temp", help="Target feature attribute column name to inject noise & recover")
    parser.add_argument("--sigma", type=float, default=0.3, help="Gaussian noise scale factor σ (default: 0.3)")
    parser.add_argument("--op", type=float, default=15.0, help="Outlier Percentage OP % (default: 15.0)")
    parser.add_argument("--method", type=str, choices=["base", "pearson", "distance", "shallow_nn", "all"], default="all", help="FID Recovery method")

    args = parser.parse_args()

    print("==========================================================================")
    print("      ENHANCED FUZZY INFORMATION DECOMPOSITION (FID) DATA RECOVERY        ")
    print("==========================================================================")

    # Load dataset
    if args.csv == "forestfires":
        if os.path.exists("forestfires.csv"):
            df = pd.read_csv("forestfires.csv")
        else:
            print("[*] Generating synthetic benchmark ForestFires dataset...")
            df = generate_sample_forestfires()
        dataset_name = "forestfires"
    elif os.path.exists(args.csv):
        df = pd.read_csv(args.csv)
        dataset_name = os.path.splitext(os.path.basename(args.csv))[0]
    else:
        print(f"[!] File not found: {args.csv}. Generating synthetic benchmark dataset...")
        df = generate_sample_forestfires()
        dataset_name = "synthetic"

    print(f"Dataset Loaded: {dataset_name} | Total Rows: {len(df)} | Columns: {list(df.columns)}")

    if args.feature not in df.columns:
        print(f"[!] Error: Target feature '{args.feature}' not found in dataset columns.")
        return

    # Preprocessing & Noise Injection
    preprocessor = EnhancedDataPreprocessor(
        df=df,
        dataset_name=dataset_name,
        sel_feature=args.feature,
        sigma=args.sigma,
        OP=args.op
    )
    
    X_scaled, X_train, X_test, y_train, y_test, features = preprocessor.prepare_data()
    feature_index = features.index(args.feature)
    compromised_indices = np.where(y_test == 0)[0]
    original_data = X_scaled[:, feature_index]

    print(f"\n[*] Target Feature: '{args.feature}' (Index: {feature_index})")
    print(f"[*] Noise Injection Config: Sigma σ = {args.sigma} | Outlier Percentage OP = {args.op}%")
    print(f"[*] Compromised Data Instances: {len(compromised_indices)} / {len(X_scaled)}")

    # Initialize FID Engine
    fid_engine = FIDRecovery()
    print(f"[*] Hardware Engine Detected: Device = {fid_engine.device.upper()}")

    methods_to_run = []
    if args.method == "all":
        methods_to_run = [
            ("Base FID", None),
            ("FID + Pearson", "pearson"),
            ("FID + Distance", "distance"),
            ("FID + Shallow NN", "shallow_nn")
        ]
    else:
        m_map = {
            "base": ("Base FID", None),
            "pearson": ("FID + Pearson", "pearson"),
            "distance": ("FID + Distance", "distance"),
            "shallow_nn": ("FID + Shallow NN", "shallow_nn")
        }
        methods_to_run = [m_map[args.method]]

    evaluator = RecoveryEvaluator(original_data, feature_name=args.feature)

    print("\n--------------------------------------------------------------------------")
    print("                         RECOVERY BENCHMARK RESULTS                       ")
    print("--------------------------------------------------------------------------")

    for label, m_type in methods_to_run:
        print(f"[*] Executing {label}...")
        try:
            recovered = fid_engine.FID(
                org_data=original_data,
                compromised_indices=compromised_indices,
                X=X_scaled,
                target_feature_index=feature_index,
                method=m_type
            )
            evaluator.evaluate_recovery(recovered, compromised_indices, label)
        except Exception as e:
            print(f"    [!] Error running {label}: {e}")

    report_df = evaluator.generate_report()
    print("\n", report_df.to_string(index=False))
    print("==========================================================================")

if __name__ == "__main__":
    main()
