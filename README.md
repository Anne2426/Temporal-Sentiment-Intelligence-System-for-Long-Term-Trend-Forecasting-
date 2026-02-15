# Temporal Sentiment Intelligence System for Long-Term Trend Forecasting


## Project Overview
This project analyzes temporal sentiment data and forecasts long-term trends using both statistical (ARIMA) and deep learning (LSTM) models. It provides a web interface for interactive exploration of sentiment trends and forecasts.

## Features
- Sentiment analysis using VADER
- Time-series aggregation and visualization
- ARIMA-based trend forecasting
- LSTM-based trend forecasting
- Streamlit web app for results

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run sentiment processing: `python main.py`
3. Run ARIMA forecast: `python arima_forecast.py`
4. Run LSTM forecast: `python lstm_forecast.py`
5. Launch the web app: `streamlit run app.py`

## Run Command
To start the Streamlit app, use:

```bash
streamlit run Temporal_Sentiment_Project/app.py
```

If you are in the Temporal_Sentiment_Project directory, use:

```bash
streamlit run app.py
```

# Temporal Sentiment Intelligence App

## Overview
This Streamlit app provides advanced sentiment analysis and forecasting for temporal data. It visualizes sentiment trends, compares ARIMA and LSTM forecasts, and offers real-time sentiment prediction for user input.

## Features
- **Sentiment Data Visualization:**
  - Interactive filtering and search for sentiment text
  - Summary statistics (mean, std, min, max, etc.)
  - Downloadable sentiment dataset
- **Forecast Outputs:**
  - ARIMA and LSTM forecast charts
  - Download buttons for forecast results
- **Forecast Comparison:**
  - Side-by-side line chart comparing ARIMA and LSTM forecasts
- **Real-time Sentiment Prediction:**
  - Enter any text for instant sentiment analysis using VADER
  - Displays sentiment score and label (positive, negative, neutral)

## Usage
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   (Ensure `vaderSentiment`, `streamlit`, `pandas`, `numpy`, `scikit-learn`, `tensorflow` or `keras` are included.)

2. **Run the app:**
   ```bash
   streamlit run Temporal_Sentiment_Project/app.py
   ```

3. **Interact with the dashboard:**
   - Filter and search sentiment data
   - Download datasets and forecasts
   - Compare ARIMA and LSTM predictions
   - Enter text for real-time sentiment analysis

## File Structure
- `app.py` — Streamlit dashboard
- `processed_sentiment.csv` — Sentiment data
- `arima_forecast.csv` — ARIMA forecast results
- `lstm_forecast.csv` — LSTM forecast results
- `requirements.txt` — Python dependencies
- `README.md` — This documentation

## Advanced Customization
- Integrate your own sentiment model for real-time prediction
- Extend forecast comparison with additional models
- Customize visualizations and statistics

## Example Input for Real-time Prediction
- "The product was excellent!"
- "Service was very poor."
- "I am happy with the results."

## License
MIT License

---
For questions or improvements, contact the project maintainer.
