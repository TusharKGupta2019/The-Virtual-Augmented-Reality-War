from textblob import TextBlob
import pandas as pd

df = pd.read_csv('vrar_datset.csv')

# Create a new column for sentiment analysis
df['sentiment'] = df['review_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Classify sentiment as positive, negative, or neutral
df['sentiment_category'] = df['sentiment'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

# Visualize the sentiment distribution
import matplotlib.pyplot as plt
sentiment_counts = df['sentiment_category'].value_counts()
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', edgecolor='black')
plt.xlabel('Sentiment Category')
plt.ylabel('Frequency')
plt.title('Sentiment Distribution')
plt.show()
