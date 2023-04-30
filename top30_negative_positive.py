# Define a function to return top 30 positive and negative words 
def get_top_words_above_below_rating(df, rating_threshold, num_top_words):
    words_above = []
    words_below = []
    stop_words = set(stopwords.words('english'))
    for index, row in df.iterrows():
        if row['stars'] > rating_threshold:
            words_above += [word.lower() for word in row['text'].split() if word.lower() not in stop_words]
        else:
            words_below += [word.lower() for word in row['text'].split() if word.lower() not in stop_words]
    word_counts_above = Counter(words_above)
    word_counts_below = Counter(words_below)
    top_words_above = word_counts_above.most_common(num_top_words)
    top_words_below = word_counts_below.most_common(num_top_words)
    return top_words_above, top_words_below

# Call the function to get the top 30 positive and negative words
top_words_above, top_words_below = get_top_words_above_below_rating(df_data, 3, 30)

# Print the top 30 positive and negative words and their counts
print("Top 30 words above 3 star ratings:")
pt_above = PrettyTable(field_names=["Word", "Count"])
[pt_above.add_row(kv) for kv in top_words_above]
pt_above.align["Word"], pt_above.align["Count"] = "l", "r"
print(pt_above)

print("\nTop 30 words below 3 star ratings:")
pt_below = PrettyTable(field_names=["Word", "Count"])
[pt_below.add_row(kv) for kv in top_words_below]
pt_below.align["Word"], pt_below.align["Count"] = "l", "r"
print(pt_below)

# Extract the counts and create a histogram
counts = [count for word, count in top_words_above]
plt.hist(counts, bins=10)
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Histogram of Top 30 Positive Word Frequencies')
plt.show()

counts = [count for word, count in top_words_below]
plt.hist(counts, bins=10)
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Histogram of Top 30 Negative Word Frequencies')
plt.show()