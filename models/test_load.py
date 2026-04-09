import sys
from pathlib import Path
import joblib
import time

print("Starting test load...")
MODEL_DIR = Path(".").resolve()
files = ["nba_predictor_model.pkl", "teams.pkl", "team_features.pkl", "feature_cols.pkl"]

for f in files:
    start = time.time()
    print(f"Loading {f}...")
    try:
        data = joblib.load(MODEL_DIR / f)
        print(f"Loaded {f} in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"Error loading {f}: {e}")

print("Test load finished.")
