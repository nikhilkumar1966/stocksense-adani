import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="StockSense",
    page_icon="📈",
    layout="wide"
)

st.title("📈 StockSense — Stock Market Predictor")
st.markdown("**Predicting the next day's stock market direction using Machine Learning.**")
st.divider()

API_KEY = "SAOPJTIPH8X2DEF4"  # Yahan apni key daalo

col1, col2 = st.columns([2, 1])

with col1:
    ticker_input = st.text_input(
        "🔍 Enter Stock Ticker Symbol",
        value="ADANIENT.BSE",
        placeholder="e.g. ADANIENT.BSE, RELIANCE.BSE, AAPL, TSLA"
    )

with col2:
    st.markdown("### 💡 Examples")
    st.markdown("""
    - `ADANIENT.BSE` — Adani Enterprises  
    - `RELIANCE.BSE` — Reliance Industries  
    - `TCS.BSE` — Tata Consultancy  
    - `AAPL` — Apple Inc  
    - `TSLA` — Tesla  
    """)

predict_btn = st.button("🚀 Predict!", use_container_width=True)
st.divider()

@st.cache_data
def fetch_data(ticker):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()

    if "Time Series (Daily)" not in data:
        return None

    ts = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={"4. close": "Close"})
    df['Close'] = pd.to_numeric(df['Close'])
    df = df[['Close']]
    return df

@st.cache_data
def train_model(ticker):
    df = fetch_data(ticker)

    if df is None or len(df) < 100:
        return None, None, None, None

    df['MA_7']          = df['Close'].rolling(7).mean()
    df['MA_21']         = df['Close'].rolling(21).mean()
    df['Daily_Return']  = df['Close'].pct_change() * 100
    df['Volatility']    = df['Daily_Return'].rolling(7).std()
    delta               = df['Close'].diff()
    gain                = delta.where(delta > 0, 0).rolling(14).mean()
    loss                = -delta.where(delta < 0, 0).rolling(14).mean()
    df['RSI']           = 100 - (100 / (1 + gain/loss))
    df['MA_Cross']      = df['MA_7'] - df['MA_21']
    df['Price_vs_MA21'] = (df['Close'] - df['MA_21']) / df['MA_21'] * 100
    df['RSI_Zone']      = pd.cut(df['RSI'],
                            bins=[0,30,50,70,100],
                            labels=[0,1,2,3]).astype(float)
    df['Target']        = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()

    features = ['MA_7','MA_21','Daily_Return','Volatility',
                'RSI','MA_Cross','Price_vs_MA21','RSI_Zone']
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        min_samples_split=20, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return df, model, features, accuracy

if predict_btn or ticker_input:
    ticker = ticker_input.strip().upper()

    with st.spinner(f"⏳ Fetching data for {ticker}..."):
        df, model, features, accuracy = train_model(ticker)

    if df is None:
        st.error("❌ Stock not found! Free API allows 25 requests/day. Try: AAPL, TSLA, RELIANCE.BSE")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Trading Days", f"{len(df)}")
        col2.metric("🤖 Accuracy", f"{accuracy*100:.2f}%")
        col3.metric("📅 Data Period", "Full History")
        col4.metric("🏦 Ticker", ticker)

        st.divider()

        st.subheader("🎯 Next Day Prediction")
        latest = df[features].iloc[-1]
        prediction = model.predict([latest])[0]
        probability = model.predict_proba([latest])[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            if prediction == 1:
                st.success("### 📈 TOMORROW: UP ✅")
            else:
                st.error("### 📉 TOMORROW: DOWN ❌")
        with col2:
            st.metric("📈 UP Probability", f"{probability[1]*100:.1f}%")
        with col3:
            st.metric("📉 DOWN Probability", f"{probability[0]*100:.1f}%")

        st.caption("⚠️ Disclaimer: Educational purpose only. Not financial advice.")
        st.divider()

        st.subheader(f"📈 {ticker} — Historical Price")
        fig1, ax1 = plt.subplots(figsize=(14, 4))
        ax1.plot(df.index, df['Close'], color='royalblue', linewidth=1.5)
        ax1.set_ylabel("Price")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

        st.divider()

        st.subheader("🔍 Feature Importance")
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        colors = ['#2ecc71' if x > 0.13 else '#e74c3c'
                  for x in importance['Importance']]
        ax2.barh(importance['Feature'], importance['Importance'], color=colors)
        ax2.set_xlabel("Importance Score")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

        st.divider()

        st.subheader("📊 RSI Indicator")
        fig3, ax3 = plt.subplots(figsize=(14, 3))
        ax3.plot(df.index, df['RSI'], color='orange', linewidth=1.2)
        ax3.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax3.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax3.set_ylabel("RSI")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)

        st.divider()

        st.subheader("📋 Latest 10 Days Data")
        st.dataframe(
            df[['Close','RSI','MA_7','MA_21','Daily_Return','Volatility']]
            .tail(10).round(2),
            use_container_width=True
        )