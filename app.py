import streamlit as st
import pandas as pd
import os
# Load processed sentiment data safely

st.title("Temporal Sentiment Intelligence Output")

# Load processed sentiment data safely
sentiment_path = os.path.join(os.path.dirname(__file__), "processed_sentiment.csv")
if os.path.exists(sentiment_path):
    df = pd.read_csv(sentiment_path)
    st.subheader("Sentiment Data")
    # Interactive filtering
    search_text = st.text_input("Search sentiment text:")
    if search_text:
        filtered_df = df[df['text'].str.contains(search_text, case=False, na=False)]
        st.dataframe(filtered_df)
    else:
        st.dataframe(df)
    # Summary statistics
    st.write("**Sentiment Summary Statistics**")
    st.write(df['sentiment'].describe())
    # Download button
    st.download_button("Download Sentiment Data", df.to_csv(index=False), "sentiment_data.csv")
else:
    st.info("Sentiment data not available.")

# Optionally show ARIMA forecast output if available
arima_path = os.path.join(os.path.dirname(__file__), "arima_forecast.csv")
if os.path.exists(arima_path):
    arima_forecast = pd.read_csv(arima_path)
    st.subheader("ARIMA Forecast")
    st.line_chart(arima_forecast)
    st.download_button("Download ARIMA Forecast", arima_forecast.to_csv(index=False), "arima_forecast.csv")
else:
    st.info("ARIMA forecast not available.")

# Optionally show LSTM forecast output if available
lstm_path = os.path.join(os.path.dirname(__file__), "lstm_forecast.csv")
if os.path.exists(lstm_path):
    lstm_forecast = pd.read_csv(lstm_path)
    st.subheader("LSTM Forecast")
    st.line_chart(lstm_forecast)
    st.download_button("Download LSTM Forecast", lstm_forecast.to_csv(index=False), "lstm_forecast.csv")
else:
    st.info("LSTM forecast not available.")

# Forecast comparison
if os.path.exists(arima_path) and os.path.exists(lstm_path):
    st.subheader("Forecast Comparison")
    comparison_df = pd.DataFrame({
        'ARIMA': arima_forecast['forecast'] if 'forecast' in arima_forecast else arima_forecast.iloc[:,0],
        'LSTM': lstm_forecast['forecast'] if 'forecast' in lstm_forecast else lstm_forecast.iloc[:,0]
    })
    st.line_chart(comparison_df)

# Real-time sentiment prediction using VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
st.subheader("Real-time Sentiment Prediction")
user_text = st.text_input("Enter text for sentiment analysis:")
if user_text:
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(user_text)
    st.write(f"Predicted sentiment score: {score['compound']}")
    if score['compound'] >= 0.05:
        st.success("Positive sentiment")
    elif score['compound'] <= -0.05:
        st.error("Negative sentiment")
    else:
        st.info("Neutral sentiment")
