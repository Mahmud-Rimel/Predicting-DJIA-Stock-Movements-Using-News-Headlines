import nltk
import pandas as pd
import numpy as np
import string
import re
import random
import itertools

nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
english_stemmer = nltk.stem.SnowballStemmer('english')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import spacy

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier, SGDRegressor,LogisticRegression
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize, binarize, LabelBinarizer
from sklearn.svm import LinearSVC
from collections import defaultdict

entity = spacy.load('en_core_web_sm')


def remove_punctuation(text):
    text = re.sub("b[(')]", '',text) #remove b from each frontline text
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word.lower() not in stopwords.words('english')])
    return text

def remove_named_entities(text):
    doc = entity(text)  # entity = spacy.load('en_core_web_sm')
    processed_text = " ".join([token.text for token in doc if not token.ent_type_])
    return processed_text


def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)


# # our data is saved on cloud. If this link is not working, please try the other one.  

df = pd.read_csv('https://basilika.uni-trier.de/nextcloud/s/5WyDLRSgNvGdMCM/download?path=%2Fdatasets&files=Combined_News_DJIA.csv')

#df = 'https://raw.githubusercontent.com/Mahmud-Rimel/Predicting-DJIA-Stock-Movements-Using-News-Headlines/main/Daily%20News%20for%20Stock%20Market%20Prediction/Combined_News_DJIA.csv'

headlines = []


for row in range(0, len(df.index)):
     headlines.append(' '.join(str(x) for x in df.iloc[row, 2:27]))
     

df['headlines'] = headlines
df['headlines'] = df['headlines'].apply(remove_punctuation)
df['headlines']  = df['headlines'].apply(remove_named_entities)
df['tokens'] = df['headlines'].apply(tokenize_and_lemmatize)   

X = df['tokens']
y = df['Label']

##  Glove with 100 Dimension


import gensim.downloader as api

glove_model = api.load("glove-wiki-gigaword-100")  #100 dimension

#glove_model = api.load("glove-wiki-gigaword-300")

## Running Glove 100 Dimension .....


def get_sentence_vector(text, model, vector_size):
    words = text.split()
    sentence_vector = np.zeros(vector_size)
    num_words = 0
    
    for word in words:
        if word in model:
            num_words += 1
            sentence_vector += model[word]
    
    if num_words > 0:
        sentence_vector /= num_words
    
    return sentence_vector


#vector_size = 100  # Match the size of the GloVe embeddings
vector_size = 100 

X_vectors = []
for text in X:
    sentence_vector = get_sentence_vector(text, glove_model, vector_size)
    X_vectors.append(sentence_vector)

X_vectors = np.array(X_vectors)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression(solver= 'liblinear', random_state=42, )  #'lbfgs' max_iter=100, penalty='l2', fit_intercept=True, multi_class='ovr', n_jobs=-1
clf_rf = RandomForestClassifier(n_estimators=200, random_state=42, criterion='entropy', max_features= 20000, n_jobs=-1)
clf_nb = BernoulliNB(alpha=0.5,binarize=0.0) 
clf_svm = SVC(gamma='auto', random_state=42)
clf_gb = GradientBoostingClassifier(random_state=42, loss= 'log_loss', learning_rate=0.025, n_estimators=200, max_features = 20000)
clf_knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
clf_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, random_state=42, tol=None, n_jobs=-1)
clf_ada = AdaBoostClassifier(random_state=42)

models_config = {'LR': clf_lr, 'RF': clf_rf, 'NB': clf_nb, 'SVM': clf_svm, 'GB': clf_gb, 'KNN': clf_knn, 'SGD': clf_sgd, 'AdaB': clf_ada}

for classifier in models_config:
    models_config[classifier].fit(X_train, y_train)
    y_predict = models_config[classifier].predict(X_test)
    print(str(classifier))
    print('accuracy: ' + str(accuracy_score(y_test, y_predict, normalize=True)))
    print('f1_score (macro): ' + str(f1_score(y_test, y_predict, average='macro')))
    print('f1_score (micro): ' + str(f1_score(y_test, y_predict, average='micro')))
    print('f1_score (weighted): ' + str(f1_score(y_test, y_predict, average='weighted')))
    print('\n')


##  Glove with 300 Dimension
## Running Glove 300 Dimension .....

import gensim.downloader as api

#glove_model = api.load("glove-wiki-gigaword-100")  #100 dimension

glove_model = api.load("glove-wiki-gigaword-300")



def get_sentence_vector(text, model, vector_size):
    words = text.split()
    sentence_vector = np.zeros(vector_size)
    num_words = 0
    
    for word in words:
        if word in model:
            num_words += 1
            sentence_vector += model[word]
    
    if num_words > 0:
        sentence_vector /= num_words
    
    return sentence_vector


#vector_size = 100  # Match the size of the GloVe embeddings
vector_size = 300 

X_vectors = []
for text in X:
    sentence_vector = get_sentence_vector(text, glove_model, vector_size)
    X_vectors.append(sentence_vector)

X_vectors = np.array(X_vectors)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression(solver= 'liblinear', random_state=42, )  #'lbfgs' max_iter=100, penalty='l2', fit_intercept=True, multi_class='ovr', n_jobs=-1
clf_rf = RandomForestClassifier(n_estimators=200, random_state=42, criterion='entropy', max_features= 20000, n_jobs=-1)
clf_nb = BernoulliNB(alpha=0.5,binarize=0.0) 
clf_svm = SVC(gamma='auto', random_state=42)
clf_gb = GradientBoostingClassifier(random_state=42, loss= 'log_loss', learning_rate=0.025, n_estimators=200, max_features = 20000)
clf_knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
clf_sgd = SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, random_state=42, tol=None, n_jobs=-1)
clf_ada = AdaBoostClassifier(random_state=42)

models_config = {'LR': clf_lr, 'RF': clf_rf, 'NB': clf_nb, 'SVM': clf_svm, 'GB': clf_gb, 'KNN': clf_knn, 'SGD': clf_sgd, 'AdaB': clf_ada}

for classifier in models_config:
    models_config[classifier].fit(X_train, y_train)
    y_predict = models_config[classifier].predict(X_test)
    print(str(classifier))
    print('accuracy: ' + str(accuracy_score(y_test, y_predict, normalize=True)))
    print('f1_score (macro): ' + str(f1_score(y_test, y_predict, average='macro')))
    print('f1_score (micro): ' + str(f1_score(y_test, y_predict, average='micro')))
    print('f1_score (weighted): ' + str(f1_score(y_test, y_predict, average='weighted')))
    print('\n')

