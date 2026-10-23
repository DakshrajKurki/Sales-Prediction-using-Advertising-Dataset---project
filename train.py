"""
Sales Prediction using Advertising Dataset
Trains and compares regression models that predict sales from TV, Radio and
Newspaper advertising spend, saves the best model and evaluation plots.

Run from the project root:  python src/train.py
"""
import os
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "advertising.csv")
IMG = os.path.join(ROOT, "images")
MODELS = os.path.join(ROOT, "models")
FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"
os.makedirs(IMG, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA)
    rename = {"tv": "TV", "radio": "Radio", "newspaper": "Newspaper", "sales": "Sales"}
    df.columns = [rename.get(c.strip().lower(), c.strip()) for c in df.columns]
    return df


def eda(df):
    print("Shape:", df.shape)
    print(df.describe().round(2))

    sns.pairplot(df, x_vars=FEATURES, y_vars=TARGET, height=4, kind="reg")
    plt.savefig(os.path.join(IMG, "pairplot.png"), dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(IMG, "correlation_heatmap.png"), dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(df[TARGET], kde=True)
    plt.title("Sales Distribution")
    plt.savefig(os.path.join(IMG, "sales_distribution.png"), dpi=120, bbox_inches="tight")
    plt.close()


def evaluate(name, model, X_train, X_test, y_train, y_test, X, y):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cv = cross_val_score(model, X, y, cv=5, scoring="r2")
    return {
        "Model": name,
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "CV_R2_mean": cv.mean(),
    }


def main():
    df = load_data()
    eda(df)

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    results = pd.DataFrame(
        [evaluate(n, m, X_train, X_test, y_train, y_test, X, y) for n, m in models.items()]
    ).sort_values("R2", ascending=False)
    print("\nModel comparison:\n", results.round(4).to_string(index=False))
    results.round(4).to_csv(os.path.join(ROOT, "results", "model_comparison.csv"), index=False)

    lr = models["Linear Regression"]
    print("\nLinear Regression coefficients:")
    print(f"  Intercept: {lr.intercept_:.4f}")
    for f, c in zip(FEATURES, lr.coef_):
        print(f"  {f}: {c:.4f}")

    best_name = results.iloc[0]["Model"]
    best = models[best_name]
    joblib.dump(best, os.path.join(MODELS, "sales_model.pkl"))
    with open(os.path.join(MODELS, "model_info.json"), "w") as f:
        json.dump({"best_model": best_name, "features": FEATURES}, f, indent=2)
    print(f"\nBest model: {best_name} -> saved to models/sales_model.pkl")

    pred = best.predict(X_test)
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, pred, alpha=0.7)
    lims = [y.min(), y.max()]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title(f"Actual vs Predicted ({best_name})")
    plt.savefig(os.path.join(IMG, "actual_vs_predicted.png"), dpi=120, bbox_inches="tight")
    plt.close()

    if hasattr(best, "feature_importances_"):
        imp = pd.Series(best.feature_importances_, index=FEATURES).sort_values()
        imp.plot(kind="barh", figsize=(6, 3), title="Feature Importance")
        plt.savefig(os.path.join(IMG, "feature_importance.png"), dpi=120, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    main()
