!pip install matplotlib
import matplotlib.pyplot as plt

!pip install pandas
import pandas as pd
!pip install nltk
!pip install prettytable
from collections import Counter
from prettytable import PrettyTable
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Read in the existing JSON file
data = []
with open('reviews.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))

df_data = pd.DataFrame(data)

# Tokenize the text and count the frequency of words
words = []
stop_words = set(stopwords.words('english'))
for review in df_data['text']:
    words += [word.lower() for word in review.split() if word.lower() not in stop_words]

word_counts = Counter(words)

# Get the top 30 words and their counts
top_words = word_counts.most_common(30)

# Extract the counts and create a histogram
counts = [count for word, count in top_words]
plt.hist(counts, bins=10)
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Histogram of Top 30 Word Frequencies')
plt.show()

# Create a PrettyTable with headers of "Word" and "Count"
pt = PrettyTable(field_names=["Word", "Count"])

# Add the most common 30 words and their counts to the PrettyTable
[pt.add_row(kv) for kv in top_words]

# Set column alignment
pt.align["Word"], pt.align["Count"] = "l", "r"

# Print the PrettyTable
print(pt)

