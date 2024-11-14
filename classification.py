import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk import FreqDist
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

def extract_char_counts(text_series):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    return text_series.apply(lambda text: [text.lower().count(c) for c in alphabet]).tolist()

def extract_whitespace_counts(text_series):
    return np.array(text_series.apply(lambda text: text.count(' ')))

def extract_pos_tag_freq(text):
    word_list = word_tokenize(text)
    pos_tags = [pos for (word, pos) in pos_tag(word_list)]
    fd = FreqDist(pos_tags)
    tag_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS']
    return np.array([fd[tag] / len(word_list) for tag in tag_list]).reshape(1, -1)

def scale(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def extract_features(df, count_vectorizer, tfidf_vectorizer, count_vector=True, tfidf_vector=True, sent_len=True, char_counts=True, whitespace_counts=True, pos_tag_counts=True):
    X_list = []
    if count_vector:
        X_list.append(extract_count_vector(df['text'], count_vectorizer))
    if tfidf_vector:
        X_list.append(extract_tfidf_vector(df['text'], tfidf_vectorizer))
    if sent_len:
        X_list.append(df['text'].apply(avg_sent_len).values.reshape(-1, 1))
    if char_counts:
        X_list.append(np.array(extract_char_counts(df['text'])))
    if whitespace_counts:
        X_list.append(extract_whitespace_counts(df['text']).reshape(-1, 1))
    if pos_tag_counts:
        X_list.append(np.vstack(df['text'].apply(extract_pos_tag_freq).values))
    return np.hstack(X_list)

def train_and_evaluate(dev=True, count_vector_used=True, tfidf_vector_used=True, sent_len=True, char_counts=True, whitespace_counts=True, pos_tag_counts=True):
    # Train classifier
    count_vector = CountVectorizer(max_features=50).fit(traindf['text'])
    tf_idf_vector = TfidfVectorizer(max_features=50).fit(traindf['text'])

    X_train = extract_features(traindf, count_vector, tf_idf_vector, count_vector_used, tfidf_vector_used, sent_len, char_counts, whitespace_counts, pos_tag_counts)
    X_train_scaled = scale(X_train)
    y_train = traindf['author']

    svm = SVC().fit(X_train_scaled, y_train)

    if dev:
        # Evaluate on dev set
        X_dev = extract_features(devdf, count_vector, tf_idf_vector, count_vector_used, tfidf_vector_used, sent_len, char_counts, whitespace_counts, pos_tag_counts)
        X_dev_scaled = scale(X_dev)
        y_dev = devdf['author']

        y_pred_dev = svm.predict(X_dev_scaled)
        f1_dev = f1_score(y_dev, y_pred_dev, average='weighted')
        print(f"Evaluated on dev set with F1 score: {f1_dev}")
        print(f"Amount of features: {X_dev.shape[1]}")
        return(f1_dev)

    else:
        # Evaluate on test set
        X_test = extract_features(testdf, count_vector, tf_idf_vector, count_vector_used, tfidf_vector_used, sent_len, char_counts, whitespace_counts, pos_tag_counts)
        X_test_scaled = scale(X_test)
        y_test = testdf['author']

        y_pred_test = svm.predict(X_test_scaled)
        f1 = f1_score(y_test, y_pred_test, average='weighted')
        print(f"Evaluated on test set with F1 score: {f1}")
        print(f"Amount of features: {X_test.shape[1]}")
        return(f1)

def ablation_study():
    f1_scores = []
    f1_scores.append(train_and_evaluate())
    f1_scores.append(train_and_evaluate(count_vector_used=False))
    f1_scores.append(train_and_evaluate(tfidf_vector_used=False))
    f1_scores.append(train_and_evaluate(sent_len=False))
    f1_scores.append(train_and_evaluate(char_counts=False))
    f1_scores.append(train_and_evaluate(whitespace_counts=False))
    f1_scores.append(train_and_evaluate(pos_tag_counts=False))
    return f1_scores

labels = ['All Features', 'Count Vector', 'TF-IDF Vector', 'Sentence Length', 'Character Counts', 'Whitespace Counts', 'POS Tag Counts']
plt.bar(labels, ablation_study())
plt.title('Ablation Study')
plt.ylabel('F1 score')
plt.show()