# =============================================
# Project 5: Autocomplete and Autocorrect Data Analytics
# Dataset: English Word Frequency (Kaggle)
# Internship: Oasis Infobyte - Level 2, Project 5
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# -----------------------------------------------
# STEP 1: Load Dataset
# -----------------------------------------------
df = pd.read_csv("C:/Users/hp/OneDrive/Desktop/datasets/datasets/unigram_freq.csv")

print("Dataset loaded!")
print("Shape:", df.shape)
print(df.head(10))

# -----------------------------------------------
# STEP 2: Data Cleaning
# -----------------------------------------------
print("\nMissing values:\n", df.isnull().sum())

# Drop missing/invalid rows
df = df.dropna()
df['word'] = df['word'].astype(str)
df = df[df['word'].str.isalpha()]   # keep only alphabetic words
df = df.reset_index(drop=True)

print("\nCleaned shape:", df.shape)
print("Total unique words:", df['word'].nunique())

# -----------------------------------------------
# STEP 3: Top Frequent Words
# -----------------------------------------------
top_words = df.sort_values('count', ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x='count', y='word', data=top_words, hue='word', palette='viridis', legend=False)
plt.title('Top 20 Most Frequent English Words')
plt.xlabel('Frequency Count')
plt.ylabel('Word')
plt.tight_layout()
plt.savefig("plot1_top_words.png")
plt.show()

# -----------------------------------------------
# STEP 4: Word Length Analysis
# -----------------------------------------------
df['word_length'] = df['word'].apply(len)

plt.figure(figsize=(8, 5))
sns.histplot(df['word_length'], bins=20, kde=True, color='steelblue')
plt.title('Distribution of Word Lengths')
plt.xlabel('Word Length')
plt.ylabel('Number of Words')
plt.tight_layout()
plt.savefig("plot2_word_length_distribution.png")
plt.show()

# -----------------------------------------------
# STEP 5: Build Autocomplete Function
# -----------------------------------------------
# Use top 50,000 words for faster lookup
word_dict = dict(zip(df['word'].head(50000), df['count'].head(50000)))

def autocomplete(prefix, top_n=5):
    """Returns top N word suggestions starting with given prefix"""
    prefix = prefix.lower()
    matches = {w: c for w, c in word_dict.items() if w.startswith(prefix)}
    sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
    return sorted_matches[:top_n]

# Test autocomplete
test_prefixes = ['app', 'comp', 'data', 'pre']
print("\n=== Autocomplete Suggestions ===")
for prefix in test_prefixes:
    suggestions = autocomplete(prefix)
    print(f"\nPrefix '{prefix}':")
    for word, count in suggestions:
        print(f"  {word} (freq: {count})")

# -----------------------------------------------
# STEP 6: Build Autocorrect Function (Edit Distance)
# -----------------------------------------------
def edit_distance(word1, word2):
    """Computes Levenshtein distance between two words"""
    m, n = len(word1), len(word2)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def autocorrect(input_word, top_n=3, max_distance=2):
    """Suggests closest correct words within max edit distance"""
    input_word = input_word.lower()
    candidates = []

    # Only check words of similar length for efficiency
    similar_length_words = df[
        (df['word_length'] >= len(input_word) - max_distance) &
        (df['word_length'] <= len(input_word) + max_distance)
    ].head(20000)

    for _, row in similar_length_words.iterrows():
        dist = edit_distance(input_word, row['word'])
        if dist <= max_distance:
            candidates.append((row['word'], dist, row['count']))

    # Sort by edit distance first, then frequency
    candidates.sort(key=lambda x: (x[1], -x[2]))
    return candidates[:top_n]

# Test autocorrect
test_typos = ['recieve', 'definately', 'wich']
print("\n=== Autocorrect Suggestions ===")
for typo in test_typos:
    suggestions = autocorrect(typo)
    print(f"\nTypo '{typo}':")
    for word, dist, count in suggestions:
        print(f"  {word} (edit distance: {dist}, freq: {count})")

# -----------------------------------------------
# STEP 7: Word Frequency vs Length (Insight)
# -----------------------------------------------
plt.figure(figsize=(8, 5))
sample = df.sample(min(5000, len(df)), random_state=42)
sns.scatterplot(x='word_length', y='count', data=sample, alpha=0.4, color='coral')
plt.yscale('log')
plt.title('Word Length vs Frequency (Log Scale)')
plt.xlabel('Word Length')
plt.ylabel('Frequency Count (log)')
plt.tight_layout()
plt.savefig("plot3_length_vs_frequency.png")
plt.show()

# -----------------------------------------------
# STEP 8: Letter Frequency Analysis
# -----------------------------------------------
all_letters = ''.join(df['word'].head(10000))
letter_counts = Counter(all_letters)
letter_df = pd.DataFrame(letter_counts.items(), columns=['Letter', 'Count']).sort_values('Count', ascending=False)

plt.figure(figsize=(12, 5))
sns.barplot(x='Letter', y='Count', data=letter_df, hue='Letter', palette='mako', legend=False)
plt.title('Letter Frequency in Top 10,000 Words')
plt.xlabel('Letter')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("plot4_letter_frequency.png")
plt.show()

# -----------------------------------------------
# STEP 9: Summary
# -----------------------------------------------
print("\n=== Summary ===")
print(f"Total Words in Dataset    : {len(df)}")
print(f"Average Word Length       : {df['word_length'].mean():.2f}")
print(f"Most Frequent Word        : {df.iloc[0]['word']} ({df.iloc[0]['count']})")
print(f"Shortest Word Length      : {df['word_length'].min()}")
print(f"Longest Word Length       : {df['word_length'].max()}")

print("\nDone! All plots saved.")