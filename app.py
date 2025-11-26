import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib 
from keras.models import load_model
import plotly.graph_objects as go
import yfinance as yf

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

# ---------------------- LOAD MODEL & SCALER ----------------------
model = load_model("stock_lstm_model.h5")

scaler = joblib.load("scaler.pkl")

# ---------------------- INDICATORS ----------------------
def add_indicators(df):
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

# ---------------------- SIDEBAR MENU ----------------------
st.sidebar.title("📊 Navigation")
menu = st.sidebar.radio("Choose Mode", 
                        ["📁 Predict using CSV", 
                         "🌐 Predict using Live Data",
                         "🔮 Predict Next 30 Days",
                         "ℹ️ About"])

# ---------------------- ABOUT PAGE ----------------------
if menu == "ℹ️ About":
    st.title("ℹ️ About This Application")
    st.write("""
    This is an advanced **Stock Market Prediction Dashboard** using an LSTM model.

    ### Features
    - Upload CSV for prediction  
    - Fetch live stock data  
    - Predict next 30 days  
    - Plotly interactive charts  
    - Indicators: MA20, MA50, RSI  
    - Download results as CSV  

    Built with **Streamlit, TensorFlow, Plotly & yFinance**.
    """)
    st.stop()

# ---------------------- CSV PREDICTION ----------------------
if menu == "📁 Predict using CSV":
    st.title("📁 Stock Prediction (Upload CSV)")

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("Data Preview")
        st.write(df.head())

        column = st.selectbox("Select price column to predict", df.columns)

        data = df[[column]].values
        scaled_data = scaler.transform(data)

        # Windowing
        window_size = 60
        x_input = []
        for i in range(window_size, len(scaled_data)):
            x_input.append(scaled_data[i - window_size:i, 0])

        if len(x_input) == 0:
            st.error("Not enough data. Need at least 60 rows.")
            st.stop()

        x_input = np.array(x_input).reshape(-1, window_size, 1)

        predictions = model.predict(x_input)
        predictions = scaler.inverse_transform(predictions)

        real_prices = data[window_size:]

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=real_prices.flatten(), name="Real Price"))
        fig.add_trace(go.Scatter(y=predictions.flatten(), name="Predicted Price"))
        fig.update_layout(title="Real vs Predicted Prices", xaxis_title="Day", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        # Download Button
        pred_df = pd.DataFrame({
            "Real Price": real_prices.flatten(),
            "Predicted Price": predictions.flatten()
        })
        st.download_button("Download Predictions", pred_df.to_csv(index=False),
                           file_name="predictions.csv")

# ---------------------- LIVE DATA PREDICTION ----------------------
if menu == "🌐 Predict using Live Data":
    st.title("🌐 Predict Using Live Stock Data (No CSV Required)")

    ticker = st.text_input("Enter Stock Ticker (Example: AAPL, TSLA, TCS.NS, RELIANCE.NS)")

    if st.button("Fetch & Predict"):
        if ticker.strip() == "":
            st.error("Please enter a valid stock ticker.")
        else:
            df = yf.download(ticker, period="5y")

            if df.empty:
                st.error("Invalid ticker or no data available.")
                st.stop()

            df = add_indicators(df)

            st.subheader("Fetched Stock Data Preview")
            st.write(df.tail())

            # Choose price column
            column = st.selectbox("Select price column", ["Close", "Open", "High", "Low"])

            data = df[[column]].values
            scaled = scaler.transform(data)

            window_size = 60
            x_input = []
            for i in range(window_size, len(scaled)):
                x_input.append(scaled[i-window_size:i, 0])

            x_input = np.array(x_input).reshape(-1, window_size, 1)

            predictions = model.predict(x_input)
            predictions = scaler.inverse_transform(predictions)

            real_prices = data[window_size:]

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=real_prices.flatten(), name="Real Price"))
            fig.add_trace(go.Scatter(y=predictions.flatten(), name="Predicted Price"))
            fig.update_layout(title=f"{ticker} - Real vs Predicted Prices",
                              xaxis_title="Day", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------- NEXT 30 DAYS FORECAST ----------------------
if menu == "🔮 Predict Next 30 Days":
    st.title("🔮 Forecast Next 30 Days")

    ticker = st.text_input("Enter Stock Ticker (For Forecasting)")

    if st.button("Predict 30 Days"):
        df = yf.download(ticker, period="5y")

        if len(df) < 60:
            st.error("Need at least 60 days of data!")
            st.stop()

        last_60 = df["Close"].values[-60:].reshape(-1, 1)
        scaled_60 = scaler.transform(last_60)

        seq = scaled_60.reshape(1, 60, 1)

        future = []
        for _ in range(30):
            pred = model.predict(seq)
            future.append(pred[0][0])
            seq = np.append(seq[:, 1:, :], [[pred]], axis=1)

        future = scaler.inverse_transform(np.array(future).reshape(-1, 1))

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=future.flatten(), mode="lines+markers", name="Forecast"))
        fig.update_layout(title=f"{ticker} - Next 30 Days Forecast",
                          xaxis_title="Day", yaxis_title="Predicted Price")

        st.plotly_chart(fig, use_container_width=True)

        # Download file
        forecast_df = pd.DataFrame({"Forecasted Price": future.flatten()})
        st.download_button("Download Forecast CSV", 
                           forecast_df.to_csv(index=False),
                           file_name="next_30_days_forecast.csv")
