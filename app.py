import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="StockSense — Adani Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 StockSense — Adani Enterprises Stock Predictor")
st.markdown("**Predicting the next day's stock market direction using Machine Learning.**")
st.divider()

@st.cache_data
def load_and_train():
    ticker = "ADANIENT.NS"
    stock = yf.download(ticker, start="2020-01-01", end="2024-12-31")
    df = stock[['Close']].copy()
    df.columns = ['Close']
    df = df.dropna()

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

with st.spinner("⏳ Fetching Adani data and training model..."):
    df, model, features, accuracy = load_and_train()

col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 Total Trading Days", f"{len(df)}")
col2.metric("🤖 Model Accuracy", f"{accuracy*100:.2f}%")
col3.metric("📅 Data Period", "2020 - 2024")
col4.metric("🏦 Stock", "ADANIENT.NS")

st.divider()

st.subheader("🎯 Next Day Prediction")

latest = df[features].iloc[-1]
prediction = model.predict([latest])[0]
probability = model.predict_proba([latest])[0]

col1, col2 = st.columns(2)

with col1:
    if prediction == 1:
        st.success("📈 ADANI KAL UPAR JAAYEGA — UP ✅")
    else:
        st.error("📉 ADANI KAL NEECHE JAAYEGA — DOWN ❌")

with col2:
    st.metric("UP Probability",   f"{probability[1]*100:.1f}%")
    st.metric("DOWN Probability", f"{probability[0]*100:.1f}%")

st.caption("⚠️ Disclaimer: Educational purpose only. Not financial advice.")
st.divider()

st.subheader("📈 Adani Historical Price (2020-2024)")
fig1, ax1 = plt.subplots(figsize=(14, 4))
ax1.plot(df.index, df['Close'], color='royalblue', linewidth=1.5)
ax1.set_ylabel("Price (INR)")
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

st.divider()

st.subheader("🔍 Feature Importance")
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

fig2, ax2 = plt.subplots(figsize=(10, 4))
colors = ['#2ecc71' if x > 0.13 else '#e74c3c' for x in importance['Importance']]
ax2.barh(importance['Feature'], importance['Importance'], color=colors)
ax2.set_xlabel("Importance Score")
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

st.divider()

st.subheader("📋 Latest 10 Days Data")
st.dataframe(df[['Close','RSI','MA_7','MA_21',
                  'Daily_Return','Volatility']].tail(10).round(2))