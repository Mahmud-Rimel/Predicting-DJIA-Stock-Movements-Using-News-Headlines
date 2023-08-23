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
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize, binarize, LabelBinarizer
from sklearn.svm import LinearSVC
from collections import defaultdict

entity = spacy.load('en_core_web_sm')
#entity = spacy.load('en_core_web_lg') # for case insensitive


## TF-IDF
### TF-IDF with ngram range(2,2) and Max features 20000
# Tf-idf vectorizer with ngram range(2,2)

class TextClassifier:
    def __init__(self, model_type='logistic_regression', max_features = 20000):
        self.model_type = model_type
        self.max_features = max_features
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.tokenize_and_lemmatize, lowercase=False, max_features=self.max_features, ngram_range= (2,2))
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(solver = 'liblinear',random_state=42)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=200, criterion='entropy')
        elif self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(criterion= 'gini',splitter='best', max_depth = 100)
        elif self.model_type == 'svm':
            self.model = svm.SVC(kernel = 'rbf')   
        elif self.model_type == 'Naive Bayes':
            self.model = BernoulliNB(alpha=0.5,binarize=0.0)    
        elif self.model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(random_state=42)
        elif self.model_type == 'stochastic_gradient':
            self.model = SGDClassifier(loss='modified_huber', random_state=0, shuffle=True)
        elif self.model_type == 'adaboost':
            self.model = AdaBoostClassifier(random_state=42)
    
    #def remove_punctuation(self, text):
        #text = re.sub("b[(')]", '', text) # remove b from each frontline text
    #    return text.translate(str.maketrans('', '', string.punctuation))
    
    

    def remove_punctuation(self, text):
        text = re.sub("b[(')]", '',text) #remove b from each frontline text
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join([word for word in text.split() if word.lower() not in stopwords.words('english')])
        return text

    def remove_named_entities(self,text):
        doc = entity(text)  # entity = spacy.load('en_core_web_sm')
        processed_text = " ".join([token.text for token in doc if not token.ent_type_])
        return processed_text
    

    def tokenize_and_lemmatize(self, text):
        tokens = word_tokenize(text.lower())
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

    
    # def tokenize_and_lemmatize(self, tokens_list):
        # lemmatizer = WordNetLemmatizer()
        # lemmatized_tokens = []
        # for tokens in tokens_list:
            # lemmatized_tokens.append([lemmatizer.lemmatize(token.lower()) for token in tokens])
        # return lemmatized_tokens

    def load_data(self, url):
        df = pd.read_csv(url)
        headlines = []
        for row in range(0, len(df.index)):
            headlines.append(' '.join(str(x) for x in df.iloc[row, 2:27]))
            

        df['headlines'] = headlines
        df['headlines'] = df['headlines'].apply(self.remove_punctuation)
        df['headlines']  = df['headlines'].apply(self.remove_named_entities)
        df['tokens'] = df['headlines'].apply(self.tokenize_and_lemmatize)
        return df
    
    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        self.model.fit(X_train_tfidf, y_train)
    
    def predict(self, X_test):
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_tfidf)
        return y_pred
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        #score = roc_auc_score(y_test, y_pred)
        score = accuracy_score(y_test, y_pred)
        return score
    

def evaluate_models(url):
    df = TextClassifier().load_data(url)

    X_train, X_test, y_train, y_test = TextClassifier().train_test_split(df['tokens'], df['Label'])

    models = {
        'logistic_regression': TextClassifier(model_type='logistic_regression', max_features = 20000),
        'random_forest': TextClassifier(model_type='random_forest', max_features = 20000),
        'decision_tree': TextClassifier(model_type='decision_tree', max_features = 20000),
        'svm': TextClassifier(model_type='svm', max_features = 20000),
        'Naive Bayes': TextClassifier(model_type='Naive Bayes', max_features = 20000),
        'gradient_boost': TextClassifier(model_type='gradient_boost', max_features = 20000), 
        'stochastic_gradient': TextClassifier(model_type='stochastic_gradient', max_features = 20000),
        'adaboost': TextClassifier(model_type='adaboost', max_features = 20000)
        
    }

    for model_type, clf in models.items():
        clf.train(X_train, y_train)
        score = clf.evaluate(X_test, y_test)
        print(f"Accuracy Score {model_type.capitalize()}: {score}")




print('Classifier Running ....')
print('Tf-idf Vectorizer with ngram range (2,2) and maximum features 20000....')

# our data is saved on cloud. If this link is not working, please try the other one. 

url = 'https://basilika.uni-trier.de/nextcloud/s/5WyDLRSgNvGdMCM/download?path=%2Fdatasets&files=Combined_News_DJIA.csv'

# url = 'https://raw.githubusercontent.com/Mahmud-Rimel/Predicting-DJIA-Stock-Movements-Using-News-Headlines/main/Daily%20News%20for%20Stock%20Market%20Prediction/Combined_News_DJIA.csv'

Result = evaluate_models(url)

print('\n')



#--------------------------------------------------------------------------
### Countvectorizer with ngram (2,2) and 20000 features
# Count vectorizer with ngram range(2,2)

class TextClassifier:
    def __init__(self, model_type='logistic_regression', max_features = 20000):
        self.model_type = model_type
        self.max_features = max_features
        self.count_vectorizer = CountVectorizer(max_features=self.max_features, ngram_range=(2,2))     #CountVectorizer
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(solver = 'liblinear',random_state=42)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=200, criterion='entropy')
        elif self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(criterion= 'gini',splitter='best', max_depth = 100)
        elif self.model_type == 'svm':
            self.model = svm.SVC(kernel = 'rbf')  
        elif self.model_type == 'Naive Bayes':
            self.model = BernoulliNB(alpha=0.5,binarize=0.0)        
        elif self.model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(random_state=42)
        elif self.model_type == 'stochastic_gradient':
            self.model = SGDClassifier(loss='modified_huber', random_state=0, shuffle=True)
        elif self.model_type == 'adaboost':
            self.model = AdaBoostClassifier(random_state=42)
    
    # def remove_punctuation(self, text):
        # text = re.sub("b[(')]", '', text) # remove b from each frontline text
        # return text.translate(str.maketrans('', '', string.punctuation))
    # 
    def remove_punctuation(self, text):
        text = re.sub("b[(')]", '',text) #remove b from each frontline text
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join([word for word in text.split() if word.lower() not in stopwords.words('english')])
        return text

    

    def remove_named_entities(self,text):
        doc = entity(text)
        processed_text = " ".join([token.text for token in doc if not token.ent_type_])
        return processed_text
        


    def tokenize_and_lemmatize(self, text):
        tokens = word_tokenize(text.lower())
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)


    def load_data(self, url):
        df = pd.read_csv(url)
        headlines = []
        for row in range(0, len(df.index)):
            headlines.append(' '.join(str(x) for x in df.iloc[row, 2:27]))
        df['headlines'] = headlines
        df['headlines'] = df['headlines'].apply(self.remove_punctuation)
        df['headlines']  = df['headlines'].apply(self.remove_named_entities)
        df['tokens'] = df['headlines'].apply(self.tokenize_and_lemmatize)
        return df
    
    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        X_train_count = self.count_vectorizer.fit_transform(X_train)
        self.model.fit(X_train_count, y_train)

    
    def predict(self, X_test):
        X_test_count = self.count_vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_count)
        return y_pred
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        #score = roc_auc_score(y_test, y_pred) # auc score
        score = accuracy_score(y_test, y_pred)
        return score

print('Classifier Running ....')
print('Count Vectorizer with ngram range (2,2) and maximum features 20000....')


# our data is saved on cloud. If this link is not working, please try the other one. 

url = 'https://basilika.uni-trier.de/nextcloud/s/5WyDLRSgNvGdMCM/download?path=%2Fdatasets&files=Combined_News_DJIA.csv'

## url = 'https://raw.githubusercontent.com/Mahmud-Rimel/Predicting-DJIA-Stock-Movements-Using-News-Headlines/main/Daily%20News%20for%20Stock%20Market%20Prediction/Combined_News_DJIA.csv'


def evaluate_models(url):
    df = TextClassifier().load_data(url)

    X_train, X_test, y_train, y_test = TextClassifier().train_test_split(df['tokens'], df['Label'])

    models = {
        'logistic_regression': TextClassifier(model_type='logistic_regression', max_features = 20000),
        'random_forest': TextClassifier(model_type='random_forest', max_features = 20000),
        'decision_tree': TextClassifier(model_type='decision_tree', max_features = 20000),
        'svm': TextClassifier(model_type='svm', max_features = 20000),
        'Naive Bayes': TextClassifier(model_type='Naive Bayes', max_features = 20000),
        'gradient_boost': TextClassifier(model_type='gradient_boost', max_features = 20000), 
        'stochastic_gradient': TextClassifier(model_type='stochastic_gradient', max_features = 20000),
        'adaboost': TextClassifier(model_type='adaboost', max_features = 20000)
        
    }

    for model_type, clf in models.items():
        clf.train(X_train, y_train)
        score = clf.evaluate(X_test, y_test)
        print(f"Accuracy Score {model_type.capitalize()}: {score}")

Result = evaluate_models(url)
print('\n')
