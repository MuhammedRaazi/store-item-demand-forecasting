import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load artifacts
model = joblib.load("xgb_model.pkl")
FEATURES = joblib.load("features.pkl")
data = joblib.load("processed_data.pkl")
data.to_csv("processed_data.csv", index=False)

st.set_page_config(page_title="Demand Forecasting", layout="wide")

st.title("📦 Store–Item Demand Forecasting")

st.markdown(
"""
**Objective:** Predict next-day sales using historical demand patterns.  
**Note:** Predictions are relative to the last available date in the dataset.
"""
)

# Sidebar
store = st.sidebar.selectbox("Select Store", sorted(data["store"].unique()))
item = st.sidebar.selectbox("Select Item", sorted(data["item"].unique()))

subset = data[(data["store"] == store) & (data["item"] == item)].sort_values("date")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Historical Sales", "Model Performance", "Tomorrow Forecast"]
)

# ------------------ Overview ------------------
with tab1:
    st.write("### Dataset Info")
    st.write(f"Date range: {subset['date'].min().date()} → {subset['date'].max().date()}")
    st.write(f"Total records: {len(subset)}")

# ------------------ Historical Sales ------------------
with tab2:
    st.write("### Actual Sales Trend")
    fig, ax = plt.subplots()
    ax.plot(subset["date"], subset["sales"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    st.pyplot(fig)

# ------------------ Model Performance ------------------
with tab3:
    st.metric("MAE", "≈ 6.15")
    st.metric("RMSE", "≈ 8.00")
    st.metric("R² Score", "≈ 0.94")

    st.markdown("""
    **Interpretation:**
    - Average prediction error ≈ 6 units
    - Model explains ~94% of variance
    """)

# ------------------ Tomorrow Forecast ------------------
with tab4:
    latest = subset.iloc[-1:].copy()

    tomorrow = latest.copy()
    tomorrow["date"] = tomorrow["date"] + pd.Timedelta(days=1)

    X_tomorrow = tomorrow[FEATURES]
    prediction = model.predict(X_tomorrow)[0]

    st.success(f"📈 Predicted sales for next day: **{prediction:.0f} units**")

    st.caption(
        "Prediction is based on last available historical data. "
        "For real-world use, daily data updates are required."
    )
