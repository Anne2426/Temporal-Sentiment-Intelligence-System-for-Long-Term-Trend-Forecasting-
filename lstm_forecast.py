
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Any

try:
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense  # type: ignore
except ImportError:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense

def load_data(filepath: str) -> np.ndarray:
    """Load sentiment data from CSV and return values as numpy array."""
    try:
        df = pd.read_csv(filepath)
        return df['sentiment'].values.reshape(-1, 1)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def prepare_sequences(data: np.ndarray, window_size: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Prepare input and output sequences for LSTM."""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape: tuple[int, int]) -> Sequential:
    """Build and compile LSTM model."""
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def forecast(model: Sequential, last_window: np.ndarray, steps: int = 5) -> np.ndarray:
    """Forecast next steps using trained model."""
    preds = []
    current = last_window.copy()
    for _ in range(steps):
        pred = model.predict(current.reshape(1, current.shape[0], 1), verbose=0)
        preds.append(pred[0, 0])
        current = np.vstack([current[1:], pred])
    return np.array(preds)

def main():
    """Main function to run LSTM forecasting."""
    values = load_data('processed_sentiment.csv')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    X, y = prepare_sequences(scaled, window_size=2)
    model = build_lstm_model((2, 1))
    model.fit(X, y, epochs=10, verbose=1)
    print("LSTM model trained successfully!")
    last_window = scaled[-2:]
    preds = forecast(model, last_window, steps=5)
    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    lstm_forecast_df = pd.DataFrame({'forecast': preds_inv})
    lstm_forecast_df.to_csv('lstm_forecast.csv', index=False)

if __name__ == "__main__":
    main()
