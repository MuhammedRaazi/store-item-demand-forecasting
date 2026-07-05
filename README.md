# Store–Item Demand Forecasting

**Machine learning system that predicts next-day sales for individual store–item pairs, achieving an R² of 0.93 and MAE of ~6 units — packaged as an interactive Streamlit app.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)

---

## Overview

Retailers lose revenue to stockouts and tie up capital in overstock when demand forecasts are inaccurate. This project builds an end-to-end forecasting pipeline that predicts **tomorrow's sales quantity** for any store–item combination using historical daily sales data — from raw data through feature engineering, model training, evaluation, and deployment in an interactive web app.

## Key Results

| Metric | Score |
|---|---|
| Test MAE | ~6 units |
| R² Score | ~0.93 |

The model substantially outperforms a naïve baseline (predicting yesterday's sales), demonstrating that engineered temporal features capture real seasonality and trend signals.

## Tech Stack

**Python** · **XGBoost** · **pandas** · **scikit-learn** · **Streamlit**

## How It Works

**1. Feature Engineering** — Raw daily sales are enriched with signals that let a tree-based model learn temporal patterns:
- Lag features (previous-day sales)
- Rolling means over multiple windows
- Growth rates to capture momentum
- Calendar features (day of week, week, month, weekend flag)

**2. Modeling** — An **XGBoost Regressor** is trained with a strictly **time-based train/test split**, ensuring no data leakage — the model is only ever evaluated on dates it hasn't seen.

**3. Evaluation** — Performance is measured with MAE, RMSE, and R², benchmarked against a naïve baseline.

**4. Deployment** — A Streamlit app lets users select any Store ID and Item ID, explore its historical sales, and view the predicted next-day sales alongside model performance metrics.

> Note: predictions are relative to the last available date in the dataset.

## Demo

### Overview
![Overview](screenshots/overview.png)

### Historical Sales
![Sales](screenshots/historical_sales.png)

### Model Performance
![Performance](screenshots/model_performance.png)

### Next-Day Prediction
![Prediction](screenshots/prediction.png)

## Run It Locally

```bash
# Clone the repo
git clone https://github.com/MuhammedRaazi/store-item-demand-forecasting.git
cd store-item-demand-forecasting

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

## Dataset

Historical daily sales data with the following schema:

| Column | Description |
|---|---|
| `date` | Sales date |
| `store_id` | Store identifier |
| `item_id` | Item identifier |
| `sales` | Units sold |

## What I Learned

- Designing lag and rolling features for time-series problems without leaking future information
- Why time-based splits matter more than random splits for forecasting
- Translating a trained model into a usable product with Streamlit
