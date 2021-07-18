import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('news.csv')
# print(df.shape, df.head(), df.columns)
# (6335, 4) Index(['Unnamed: 0', 'title', 'text', 'label'], dtype='object')
x = df['text']
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=7)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter=50)

pac.fit(tfidf_train,y_train)

prediction = pac.predict(tfidf_test)

score = accuracy_score(y_test, prediction)

print(f'Accuracy: {round(score*100, 2)}%\n')


con_matrix = confusion_matrix(y_test, prediction, labels=['FAKE', 'REAL'])

print(f'True Positive:\t{con_matrix[0][0]}')
print(f'False Positive:\t{con_matrix[0][1]}')
print(f'True Negative:\t{con_matrix[1][0]}')
print(f'False Negative:\t{con_matrix[1][1]}')