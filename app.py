import streamlit as st
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
import plotly.graph_objects as go

# Load Model & Scaler
model = load_model("stock_lstm_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Stock Prediction", layout="wide")

# Sidebar
st.sidebar.title("📊 Options")
mode = st.sidebar.radio("Select Mode", ["Predict CSV", "Predict Next 30 Days", "About"])

# ---------------------- ABOUT PAGE -----------------------------

if mode == "About":
    st.title("📘 About This App")
    st.write("""
    This Stock Prediction Dashboard uses a trained **LSTM model** to forecast stock prices.
    
    ### 🔧 Features
    - Upload CSV & predict prices  
    - Predict next 30 days  
    - Moving Averages (MA20, MA50)  
    - RSI Indicator  
    - Interactive charts  
    - Downloadable prediction file  
    """)
    st.stop()

# ---------------------- CSV PREDICTION -----------------------------

if mode == "Predict CSV":
    st.title("📈 Stock Market Prediction (Upload CSV)")

    uploaded = st.file_uploader("Upload a stock CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("Preview of Uploaded Data")
        st.write(df.head())

        column = st.selectbox("Select price column to predict", df.columns)

        data = df[[column]].values
        scaled_data = scaler.transform(data)

        window_size = 60
        x_input = []
        for i in range(window_size, len(scaled_data)):
            x_input.append(scaled_data[i-window_size:i, 0])

        if len(x_input) == 0:
            st.error("Need at least 60 rows of data!")
            st.stop()

        x_input = np.array(x_input).reshape(-1, window_size, 1)

        predictions = model.predict(x_input)
        predictions = scaler.inverse_transform(predictions)

        real_prices = data[window_size:]

        # Plot with Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=real_prices.flatten(), mode='lines', name="Real Price"))
        fig.add_trace(go.Scatter(y=predictions.flatten(), mode='lines', name="Predicted Price"))
        fig.update_layout(title="Real vs Predicted Stock Price", xaxis_title="Days", yaxis_title="Price")

        st.plotly_chart(fig, use_container_width=True)

        # Download predictions
        pred_df = pd.DataFrame({
            "Real Price": real_prices.flatten(),
            "Predicted Price": predictions.flatten()
        })
        st.download_button("Download Prediction CSV", pred_df.to_csv(index=False),
                           file_name="predictions.csv")

# ---------------------- NEXT 30 DAYS FORECAST -----------------------------

if mode == "Predict Next 30 Days":
    st.title("🔮 Predict Next 30 Days")

    uploaded = st.file_uploader("Upload last 60 days closing price CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        column = st.selectbox("Select target column", df.columns)

        if len(df) < 60:
            st.error("Need exactly 60 days input for forecasting!")
            st.stop()

        last_60 = df[[column]].values[-60:]
        scaled_60 = scaler.transform(last_60)

        future_input = scaled_60.reshape(1, 60, 1)

        next_30 = []
        for _ in range(30):
            pred = model.predict(future_input)
            next_30.append(pred[0][0])
            future_input = np.append(future_input[:, 1:, :], [[pred]], axis=1)

        next_30 = scaler.inverse_transform(np.array(next_30).reshape(-1, 1))

        # Plot forecast
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=next_30.flatten(), mode='lines+markers', name="Forecast"))
        fig2.update_layout(title="Next 30 Days Forecast", xaxis_title="Days", yaxis_title="Price")

        st.plotly_chart(fig2, use_container_width=True)

        # Download
        forecast_df = pd.DataFrame({"Forecasted Price": next_30.flatten()})
        st.download_button("Download Forecast CSV", forecast_df.to_csv(index=False),
                           file_name="next_30_days_prediction.csv")
