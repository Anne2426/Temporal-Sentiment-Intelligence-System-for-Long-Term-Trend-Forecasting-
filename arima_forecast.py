
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

df = pd.read_csv('processed_sentiment.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

model = ARIMA(df['sentiment'], order=(1,1,1))
model_fit = model.fit()

# Forecast next 5 steps
forecast = model_fit.forecast(steps=5)

# Save forecast to CSV for web app
forecast_df = forecast.reset_index(drop=True).to_frame(name='forecast')
forecast_df.to_csv('arima_forecast.csv', index=False)

print("Future Sentiment Forecast:")
print(forecast)


# Optionally plot
forecast.plot(title="ARIMA Sentiment Forecast")
plt.show()
