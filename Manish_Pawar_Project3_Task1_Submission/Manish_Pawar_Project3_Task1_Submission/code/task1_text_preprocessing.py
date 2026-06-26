# Project 3 - Task 1: Text Preprocessing and Feature Extraction
# Name: Manish Pawar

import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

reviews = [
    ('This movie was absolutely fantastic! Best film of the year.', 'positive'),
    ('Terrible acting and boring plot. Complete waste of time!', 'negative'),
    ('Loved every minute of it. The director did an amazing job.', 'positive'),
    ('Awful storyline. I fell asleep halfway through.', 'negative'),
    ('Brilliant performances! The cinematography was breathtaking.', 'positive'),
    ('Worst movie I have ever seen. Do not waste your money.', 'negative'),
    ('Heartwarming and beautifully crafted. A true masterpiece.', 'positive'),
    ('Dull and predictable. The script was painfully bad.', 'negative'),
]

df = pd.DataFrame(reviews, columns=['review', 'sentiment'])

STOP_WORDS = set("a an and are as at be been being by for from has have he her hers him his i in into is it its itself me my of on or our ours she that the their theirs them they this to was we were with you your yours do does did not no nor so than then very all any can will just should now only over once here there when where why how what which who whom if because while during again further few more most other some such own same too movie film".split())
LEMMA_MAP = {'performances':'performance','scenes':'scene','characters':'character','visuals':'visual','actors':'actor','jokes':'joke','loved':'love','amazing':'amaze'}

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)           # remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)      # remove punctuation and numbers
    return re.sub(r'\s+', ' ', text).lower().strip()

def tokenize_text(text):
    return re.findall(r'[a-zA-Z]+', text.lower())

def lemmatize(word):
    return LEMMA_MAP.get(word, word)

def preprocess_text(text):
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [lemmatize(t) for t in tokens]
    return ' '.join(tokens)

df['cleaned_review'] = df['review'].apply(preprocess_text)

sample = 'The movie was AMAZING!!! Best film of 2024. Visit www.movies.com'
print('Original:', sample)
print('Cleaned :', clean_text(sample))
print('Tokens  :', tokenize_text(clean_text(sample)))

for i in range(len(df)):
    print('\nOriginal:', df.loc[i, 'review'])
    print('Cleaned :', df.loc[i, 'cleaned_review'])

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1, max_df=0.95)
X = vectorizer.fit_transform(df['cleaned_review'])
print('\nShape of TF-IDF matrix:', X.shape)
print('Vocabulary size:', len(vectorizer.vocabulary_))
print('Sample feature names:', vectorizer.get_feature_names_out()[:10])

def get_top_words(df_subset, n=15):
    all_words = ' '.join(df_subset['cleaned_review']).split()
    return Counter(all_words).most_common(n)

pos_words = get_top_words(df[df['sentiment'] == 'positive'])
neg_words = get_top_words(df[df['sentiment'] == 'negative'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, words, title in [(axes[0], pos_words, 'Top Words in POSITIVE Reviews'),
                         (axes[1], neg_words, 'Top Words in NEGATIVE Reviews')]:
    ws = [w[0] for w in words]
    cs = [w[1] for w in words]
    ax.barh(ws[::-1], cs[::-1])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Frequency')
plt.tight_layout()
plt.savefig('top_words_by_sentiment.png', dpi=150)
plt.show()
