import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import StandardScaler

devdf = pd.read_csv('pan2425_dev_data.csv')
testdf = pd.read_csv('pan2425_test_data.csv')
traindf = pd.read_csv('pan2425_train_data.csv')

def extract_count_vector(text, count_vectorizer):
    return count_vectorizer.transform(text).toarray()

def extract_tfidf_vector(text, tfidf_vectorizer):
    return tfidf_vectorizer.transform(text).toarray()

def avg_sent_len(text):
    sent_list = sent_tokenize(text)
    return sum(len(sent) for sent in sent_list)/len(sent_list)

def character_counts(text_series):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    return text_series.apply(lambda text: [text.lower().count(c) for c in alphabet]).tolist()

def scale(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def extract_features(df, count_vectorizer, tfidf_vectorizer):
    count_vector = extract_count_vector(df['text'], count_vectorizer)
    tfidf_vector = extract_tfidf_vector(df['text'], tfidf_vectorizer)
    sent_len = df['text'].apply(avg_sent_len)
    char_counts = character_counts(df['text'])
    return np.hstack((count_vector, tfidf_vector, sent_len.values.reshape(-1, 1), char_counts))

count_vector = CountVectorizer(max_features=50).fit(traindf['text'])
tf_idf_vector = TfidfVectorizer(max_features=50).fit(traindf['text'])

X_train = extract_features(traindf, count_vector, tf_idf_vector)
X_train_scaled = scale(X_train)
y_train = traindf['author']

svm = SVC().fit(X_train_scaled, y_train)

X_test = extract_features(testdf, count_vector, tf_idf_vector)
X_test_scaled = scale(X_test)
y_test = testdf['author']

y_pred = svm.predict(X_test_scaled)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'F1 Score: {f1}')
print(f'With number of features: {X_train.shape[1]}')

