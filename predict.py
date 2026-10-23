"""
Predict sales for a given advertising budget using the saved model.

Usage:  python src/predict.py --tv 150 --radio 25 --newspaper 20
"""
import argparse
import os

import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "sales_model.pkl")


def main():
    p = argparse.ArgumentParser(description="Predict sales from ad spend")
    p.add_argument("--tv", type=float, required=True, help="TV ad spend (in thousands)")
    p.add_argument("--radio", type=float, required=True, help="Radio ad spend (in thousands)")
    p.add_argument("--newspaper", type=float, required=True, help="Newspaper ad spend (in thousands)")
    a = p.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise SystemExit("Model not found. Run `python src/train.py` first.")

    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([[a.tv, a.radio, a.newspaper]], columns=["TV", "Radio", "Newspaper"])
    print(f"Predicted Sales: {model.predict(X)[0]:.2f} (thousand units)")


if __name__ == "__main__":
    main()
