
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

df = pd.read_csv('sentiment_data.csv')
df['date'] = pd.to_datetime(df['date'])

sia = SentimentIntensityAnalyzer()
df['sentiment'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])
df.to_csv('processed_sentiment.csv', index=False)
print("Sentiment processing completed!")
