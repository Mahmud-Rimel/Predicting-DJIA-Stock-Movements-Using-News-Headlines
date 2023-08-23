import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk import pos_tag                     #nvad ratio
from nltk.tokenize import word_tokenize
from collections import Counter

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Readability score and more
import textstat
from textstat import smog_index, gunning_fog,  dale_chall_readability_score,flesch_kincaid_grade,coleman_liau_index
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor,LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score



# Bert Tokenizer 
from transformers import BertTokenizer, BertModel
import torch


# our data is saved on cloud. If this link is not working, please try the other one. 

df = pd.read_csv('https://basilika.uni-trier.de/nextcloud/s/5WyDLRSgNvGdMCM/download?path=%2Fdatasets&files=Combined_News_DJIA.csv')

#df = 'https://raw.githubusercontent.com/Mahmud-Rimel/Predicting-DJIA-Stock-Movements-Using-News-Headlines/main/Daily%20News%20for%20Stock%20Market%20Prediction/Combined_News_DJIA.csv'

headlines = []


for row in range(0, len(df.index)):
     headlines.append(' '.join(str(x) for x in df.iloc[row, 2:27]))


def remove_punctuation(text):
    text = re.sub("b[(')]", '', text) # remove b from each frontline text
    return text.translate(str.maketrans('', '', string.punctuation))

df['headlines'] = headlines
df['headlines'] = df['headlines'].apply(remove_punctuation)

X = df['headlines']
y = df['Label']


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embeddings

# Get BERT embeddings for each text in the dataset
X_embeddings = []
for text in X:
    X_embeddings.append(bert_embeddings(text)[0])


## Stylistic Features

# get subjectivity

def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# get polarity

def get_polarity(text):
    return TextBlob(text).sentiment.polarity    

#--------------------------------------------------------    
# Vocabulary Richness:

def yules_k(text):
    word_list = word_tokenize(text)
    word_counts = Counter(word_list)
    unique_word_count = len(word_counts)
    total_word_count = len(word_list)
    yules_k = 10_000 * (unique_word_count - total_word_count)/total_word_count
    return yules_k
    
def sichel_measure(text):
    word_list = word_tokenize(text)
    word_count = len(word_list)
    word_set = set(word_list)
    word_unique_count = len(word_set)
    
    sichel_score = word_unique_count / word_count
    
    return sichel_score

#--------------------------------------------------
# N-V-A-D ratio
def nvad_ratio(text):
    words = word_tokenize(text)
    tagged = pos_tag(words)
    
    n = 0
    v = 0
    a = 0
    d = 0
    
    for word, tag in tagged:
        if tag.startswith('N'):
            n += 1
        elif tag.startswith('V'):
            v += 1
        elif tag.startswith('J'):
            a += 1
        elif tag.startswith('R'):
            d += 1
    
    nvad = (n + v + a + d) / len(words)
    
    return nvad


#------------------------------------------------------    
# create function  to get the sentiment score:

def getSIA(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

    # get Sentiment score for each day:
compound = []
neg = []
pos = []
neu = []
SIA = 0 

for i in range(0, len(df['headlines'])):
    SIA = getSIA(df['headlines'][i])
    compound.append(SIA['compound'])
    neg.append(SIA['neg'])
    pos.append(SIA['pos'])
    neu.append(SIA['neu'])


#-------------------------------------Applying Features -----------------

df['text_len']  = df['headlines'].apply(lambda x: len(word_tokenize(x))) 
# Subjectivity
df['subjectivity'] = df['headlines'].apply(get_subjectivity)
df['polarity'] = df['headlines'].apply(get_polarity)

#Readability
df['gunny_fox'] = df['headlines'].apply(lambda x : gunning_fog(x))
df['smog_index'] = df['headlines'].apply(lambda x : smog_index(x))
df['dale_chall']  = df['headlines'].apply(lambda x: dale_chall_readability_score(x))
df['flesch_kincaid']  = df['headlines'].apply(lambda x: flesch_kincaid_grade(x))
df['coleman_liau_index'] = df['headlines'].apply(lambda x : coleman_liau_index(x))

# vocabulary richness
df['yules_k'] = df['headlines'].apply(yules_k)
df['sichel'] = df['headlines'].apply(sichel_measure)

# n-v-a-d ratio
df['nvad_ratio'] = df['headlines'].apply(nvad_ratio)

df['compound'] = compound
df['pos'] = pos
df['neg'] = neg
df['neutral'] = neu        

## Combining Stylistic Features with BERT Embedding 

# Combine BERT embeddings and text features
text_features = df[['text_len','subjectivity','gunny_fox','smog_index','dale_chall','flesch_kincaid','coleman_liau_index','yules_k','sichel', 'nvad_ratio','compound','pos','neg','neutral' ]].values

X_combined = [np.concatenate((embedding, text_feature)) for embedding, text_feature in zip(X_embeddings, text_features)]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression(solver= 'liblinear', random_state=42 )  #'lbfgs' max_iter=100, penalty='l2', fit_intercept=True, multi_class='ovr', 
n_jobs=-1
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
