import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('vrar_datset.csv')

# Calculate sentiment for each product
sentiments = df.apply(lambda x: TextBlob(x['review_text']).sentiment.polarity, axis=1)
df['sentiment'] = sentiments

# Group reviews by product name
product_reviews = df.groupby('product_name')

# Calculate the average sentiment for each product
average_sentiments = product_reviews['sentiment'].mean()

# Convert the sentiments to a DataFrame
sentiments_df = average_sentiments.to_frame('sentiment').reset_index()

# Categorize sentiments
def categorize_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

sentiments_df['sentiment_category'] = sentiments_df['sentiment'].apply(categorize_sentiment)

# Visualize the sentiment distribution with categories
plt.figure(figsize=(12, 8))

# Create bars and legends for each product
for product in sentiments_df['product_name'].unique():
    product_data = sentiments_df[sentiments_df['product_name'] == product]
    plt.bar(product_data['product_name'], product_data['sentiment'], label=product)

plt.xlabel('Product Name')
plt.ylabel('Average Sentiment')
plt.title('Product-wise Sentiment Distribution')
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
plt.tight_layout()
plt.show()
