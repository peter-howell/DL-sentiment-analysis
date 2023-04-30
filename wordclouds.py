import pandas as pd
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the review data
data = []
with open('reviews.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Create a dataframe from the review data
df_data = pd.DataFrame(data)

# Define a function to label reviews as positive or negative
def label_review(row):
    if row['stars'] > 3:
        return 'positive'
    else:
        return 'negative'

# Apply the label_review function to each row in the dataframe
df_data['label'] = df_data.apply(lambda row: label_review(row), axis=1)

# Concatenate all the positive reviews into a single string
positive_text = ' '.join(df_data[df_data['label']=='positive']['text'])

# Concatenate all the negative reviews into a single string
negative_text = ' '.join(df_data[df_data['label']=='negative']['text'])

# Generate a word cloud for positive reviews
positive_wordcloud = WordCloud(width=600, height=300, background_color='white', colormap='BuGn_r').generate(positive_text)

# Generate a word cloud for negative reviews
negative_wordcloud = WordCloud(width=600, height=300, background_color='black', colormap='hot_r').generate(negative_text)

# Plot the word clouds
fig, axs = plt.subplots(1, 2, figsize=(12, 10))
axs[0].imshow(positive_wordcloud, interpolation='bilinear')
axs[0].axis('off')
axs[0].set_title('Positive Reviews')
axs[1].imshow(negative_wordcloud, interpolation='bilinear')
axs[1].axis('off')
axs[1].set_title('Negative Reviews')
plt.save_fig('wordclouds.png')