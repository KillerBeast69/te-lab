import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK datasets (run once)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Sample Document
text = "Data science is an amazing field. Scientists are analyzing large datasets and studying patterns easily."
print("Original Text:\n", text)

# ==========================================
# PART 1: Document Preprocessing Methods
# ==========================================

# 1. Tokenization (Splitting text into individual words)
tokens = word_tokenize(text)
print("\n1. Tokenization:\n", tokens)

# 2. POS Tagging (Assigning Parts of Speech like NN=Noun, VB=Verb, JJ=Adjective)
pos_tags = nltk.pos_tag(tokens)
print("\n2. POS Tagging:\n", pos_tags)

# 3. Stop Words Removal (Removing common words like 'is', 'an', 'and' and punctuation)
stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in tokens if w.lower() not in stop_words and w.isalpha()]
print("\n3. After Stop Words Removal:\n", filtered_tokens)

# 4. Stemming (Chopping suffixes off words)
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(w) for w in filtered_tokens]
print("\n4. Stemming:\n", stemmed_words)

# 5. Lemmatization (Converting words to valid dictionary base form)
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(w.lower(), pos='v') for w in filtered_tokens]
print("\n5. Lemmatization:\n", lemmatized_words)


# ==========================================
# PART 2: Term Frequency & Inverse Document Frequency (TF-IDF)
# ==========================================
documents = [
    "Data science is fun and exciting",
    "Machine learning is a part of data science",
    "Data analytics and machine learning are popular"
]

# Calculate TF-IDF using Scikit-Learn
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)

# Convert to a neat DataFrame for display
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
print("\n--- PART 2: TF-IDF Matrix Representation ---")
print(df_tfidf.round(3))