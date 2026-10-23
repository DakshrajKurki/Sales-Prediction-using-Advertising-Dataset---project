# Sales Prediction using Advertising Dataset

Predicting product **sales** from advertising spend on **TV**, **Radio** and **Newspaper** using machine learning regression models.

## Problem Statement
A company wants to understand how its advertising budget across different channels affects sales, and to forecast sales for a planned budget. This project builds and compares regression models to make that prediction.

## Dataset
`data/advertising.csv` — 200 rows, no missing values.

| Column | Description |
|---|---|
| TV | Advertising spend on TV (in thousands) |
| Radio | Advertising spend on Radio (in thousands) |
| Newspaper | Advertising spend on Newspaper (in thousands) |
| Sales | Units sold (in thousands) — **target** |

## Project Structure
```
├── data/
│   └── advertising.csv
├── notebooks/
│   └── Sales_Prediction.ipynb     # Full EDA + modelling walkthrough
├── src/
│   ├── train.py                   # Trains & compares models, saves best one
│   └── predict.py                 # Predicts sales for a custom budget
├── models/
│   ├── sales_model.pkl            # Saved best model
│   └── model_info.json
├── images/                        # EDA and evaluation plots
├── results/
│   └── model_comparison.csv
├── requirements.txt
└── README.md
```

## Approach
1. **EDA** — distributions, pair plots and correlation analysis
2. **Train/test split** — 80/20, `random_state=42`
3. **Models** — Linear Regression, Random Forest, Gradient Boosting
4. **Evaluation** — R², MAE, RMSE and 5-fold cross-validated R²
5. **Prediction** — CLI script for new advertising budgets

## Results

| Model | R² | MAE | RMSE | CV R² |
|---|---|---|---|---|
| **Gradient Boosting** | **0.983** | **0.619** | **0.730** | **0.978** |
| Random Forest | 0.982 | 0.629 | 0.758 | 0.976 |
| Linear Regression | 0.899 | 1.461 | 1.782 | 0.887 |

**Linear Regression equation:**
`Sales = 2.979 + 0.0447·TV + 0.1892·Radio + 0.0028·Newspaper`

![Correlation](images/correlation_heatmap.png)
![Actual vs Predicted](images/actual_vs_predicted.png)
![Feature Importance](images/feature_importance.png)

## Key Insights
- **TV** is the strongest driver of sales, followed by **Radio**.
- **Newspaper** spend has almost no effect on sales.
- Tree-based models outperform linear regression because they capture the **interaction between TV and Radio** spend.
- **Recommendation:** reallocate Newspaper budget toward TV and Radio.

## How to Run
```bash
git clone https://github.com/DakshrajKurki/Sales-Prediction-using-Advertising-Dataset---project.git
cd Sales-Prediction-using-Advertising-Dataset---project
pip install -r requirements.txt

# Train models (regenerates plots, results and saved model)
python src/train.py

# Predict sales for a custom budget
python src/predict.py --tv 150 --radio 25 --newspaper 20
# -> Predicted Sales: 14.69 (thousand units)
```
Or open `notebooks/Sales_Prediction.ipynb` in Jupyter.

## Tech Stack
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter

## Author
**Dakshraj Singh Chandawat** — [GitHub](https://github.com/DakshrajKurki)
