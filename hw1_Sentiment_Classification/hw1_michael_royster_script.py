# Michael Royster - CSCI 544 HW 1
# python 3.9.13
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('universal_tagset', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
import re
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings('ignore')

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz
import requests
import contractions

url = 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz'
url_data = requests.get(url)

with open("data.tsv.gz", "wb") as f:
    f.write(url_data.content)


df = pd.read_csv("data.tsv", sep='\t', on_bad_lines="skip")
df = df[['review_body', 'star_rating']]

def categorize(x):
    if x > 3:
        return 3
    elif x < 3:
        return 1
    else:
        return 2

# Form classes
df = df[df['star_rating'].apply(lambda x: isinstance(x, int))]
df['rating_class'] = df['star_rating'].apply(lambda x: categorize(x))
df.drop(columns=['star_rating'], inplace=True)

# print(df.head())

class_one = df.query('rating_class == 1').sample(n=20000)
class_two = df.query('rating_class == 2').sample(n=20000)
class_three = df.query('rating_class == 3').sample(n=20000)

data_set = pd.concat([class_one, class_two, class_three])
data_set.reset_index(drop=True, inplace=True)
# print(data_set.head())
# print(data_set.shape[0])

def remove_html(text):
    if text == '': return text
    if not isinstance(text, str): 
        return str(text)
    if len(str(text)) < 3: return text
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def clean_non_alpha(word):
    return '' if re.search('[^a-zA-Z]', word) != None else word

before_cleaning = np.sum(data_set['review_body'].str.len())/len(data_set['review_body'])

# lowercase
data_set['review_body'] = data_set['review_body'].str.lower()
# remove HTML
data_set['review_body'] = data_set['review_body'].apply(lambda x: remove_html(x))
# remove URLs
data_set['review_body'] = data_set['review_body'].replace('http\S+','',regex=True)
# perform contractions
data_set['review_body'] = data_set['review_body'].apply(lambda x: contractions.fix(x))
# remove non-alpha characters
# data_set['review_body'].replace('[^a-zA-Z\s]',' ',regex=True, inplace=True)
data_set['review_body'] = data_set['review_body'].apply(lambda text: " ".join(clean_non_alpha(text) for text in nltk.word_tokenize(text)))
# remove extra spaces
data_set['review_body'] = data_set['review_body'].replace('\s+',' ',regex=True)

after_cleaning = np.sum(data_set['review_body'].str.len())/len(data_set['review_body'])
# print(f"Average length before and after cleaning: {before_cleaning}, {after_cleaning}")

# from nltk.corpus import stopwords

before_preproc = np.sum(data_set['review_body'].str.len())/len(data_set['review_body'])
# stop_words = set(stopwords.words('english'))
# data_set['review_body'] = data_set['review_body'].apply(lambda text: " ".join(word for word in nltk.word_tokenize(text) if word not in stop_words))

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

def get_pos(x):
    return x if x in ['n','v','a','r','s'] else 'n'

lemmatizer = WordNetLemmatizer()
data_set['review_body'] = data_set['review_body'].apply(lambda body: " ".join([lemmatizer.lemmatize(w, get_pos(p[0].lower()))for w, p in pos_tag(nltk.word_tokenize(body), tagset='universal')]))

after_preproc = np.sum(data_set['review_body'].str.len())/len(data_set['review_body'])
# print(f"Average length before and after pre-processing: {before_preproc}, {after_preproc}")

from sklearn.feature_extraction.text import TfidfVectorizer

data_set['review_body'] = data_set['review_body'].replace('', np.nan)
data_set = data_set.dropna(subset=['review_body'])
data_set.reset_index(drop=True, inplace=True)

tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2)) # try max_features, min_df, max_df 
x = tfidf.fit_transform(data_set['review_body'])

tfidf_values = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names_out())
full_set = pd.concat([data_set, tfidf_values], axis=1)

full_one = full_set.query('rating_class == 1')
full_two = full_set.query('rating_class == 2')
full_three = full_set.query('rating_class == 3')

# Calculating number of training rows
one_train_size = int(full_one.shape[0] * 0.8)
two_train_size = int(full_two.shape[0] * 0.8)
three_train_size = int(full_three.shape[0] * 0.8)

# Sampling the 80% for training
train_one = full_one.iloc[0:one_train_size, list(range(0,full_one.shape[1]))]
train_two = full_two.iloc[0:two_train_size, list(range(0,full_one.shape[1]))]
train_three = full_three.iloc[0:three_train_size, list(range(0,full_one.shape[1]))]

# Sampling the 20% for testing
test_one = full_one.iloc[one_train_size:full_one.shape[0], list(range(0,full_one.shape[1]))]
test_two = full_two.iloc[two_train_size:full_two.shape[0], list(range(0,full_one.shape[1]))]
test_three = full_three.iloc[three_train_size:full_three.shape[0], list(range(0,full_one.shape[1]))]

# Combining the training and testing population
train_population = pd.concat([train_one, train_two, train_three])
test_population = pd.concat([test_one, test_two, test_three])

# train_population.to_csv('Training.csv', index=False)
# test_population.to_csv('Testing.csv', index=False)

assert train_population.shape[0] + test_population.shape[0] == full_set.shape[0], "Rows have been missed"

from sklearn.linear_model import Perceptron
from sklearn import metrics

perceptron = Perceptron()

perceptron.fit(train_population.iloc[0:train_population.shape[0], list(range(2,train_population.shape[1]))], train_population['rating_class'])

prediction = perceptron.predict(test_population.iloc[0:test_population.shape[0], list(range(2,test_population.shape[1]))])
perceptron_results = pd.DataFrame(zip(test_population.iloc[0:test_population.shape[0], 1], prediction), columns=['Label', 'Prediction'])
score = perceptron_results.query('Label == Prediction').shape[0]

# print("Perceptron Results:")
perceptron_report = metrics.classification_report(perceptron_results['Label'], perceptron_results['Prediction'], output_dict=True)
# print(metrics.classification_report(perceptron_results['Label'], perceptron_results['Prediction']))

from sklearn import svm

machine = svm.LinearSVC()
machine.fit(train_population.iloc[0:train_population.shape[0], list(range(2,train_population.shape[1]))], train_population['rating_class'])
m_prediction = machine.predict(test_population.iloc[0:test_population.shape[0], list(range(2,test_population.shape[1]))])
machine_results = pd.DataFrame(zip(test_population.iloc[0:test_population.shape[0], 1], m_prediction), columns=['Label', 'Prediction'])

svm_report = metrics.classification_report(machine_results['Label'], machine_results['Prediction'], output_dict=True)
# print(metrics.classification_report(machine_results['Label'], machine_results['Prediction']))

from sklearn.linear_model import LogisticRegression

# regression = LogisticRegression(solver='lbfgs', multi_class='auto')
regression = LogisticRegression(max_iter=500)
regression.fit(train_population.iloc[0:train_population.shape[0], list(range(2,train_population.shape[1]))], train_population['rating_class'])
r_prediction = regression.predict(test_population.iloc[0:test_population.shape[0], list(range(2,test_population.shape[1]))])
regression_results = pd.DataFrame(zip(test_population.iloc[0:test_population.shape[0], 1], r_prediction), columns=['Label', 'Prediction'])
lr_report = metrics.classification_report(regression_results['Label'], regression_results['Prediction'], output_dict=True)
# print(metrics.classification_report(regression_results['Label'], regression_results['Prediction']))

from sklearn.naive_bayes import MultinomialNB

bayes = MultinomialNB()
bayes.fit(train_population.iloc[0:train_population.shape[0], list(range(2,train_population.shape[1]))], train_population['rating_class'])
b_prediction = bayes.predict(test_population.iloc[0:test_population.shape[0], list(range(2,test_population.shape[1]))])
bayes_results = pd.DataFrame(zip(test_population.iloc[0:test_population.shape[0], 1], b_prediction), columns=['Label', 'Prediction'])
nb_report = metrics.classification_report(bayes_results['Label'], bayes_results['Prediction'], output_dict=True)
# print(metrics.classification_report(bayes_results['Label'], bayes_results['Prediction']))

# Output
print(f"Data Cleaning avg length (before, after): {before_cleaning}, {after_cleaning}")
print(f"Data Pre-processing avg length (before, after): {before_preproc}, {after_preproc}")
print()
print("Model: (Precision, Recall, F1-Score)")
print(f"Perceptron Class 1: {perceptron_report['1']['precision']}, {perceptron_report['1']['recall']}, {perceptron_report['1']['f1-score']}")
print(f"Perceptron Class 2: {perceptron_report['2']['precision']}, {perceptron_report['2']['recall']}, {perceptron_report['2']['f1-score']}")
print(f"Perceptron Class 3: {perceptron_report['3']['precision']}, {perceptron_report['3']['recall']}, {perceptron_report['3']['f1-score']}")
print(f"Perceptron Average: {perceptron_report['weighted avg']['precision']}, {perceptron_report['weighted avg']['recall']}, {perceptron_report['weighted avg']['f1-score']}")
print()
print(f"SVM Class 1: {svm_report['1']['precision']}, {svm_report['1']['recall']}, {svm_report['1']['f1-score']}")
print(f"SVM Class 2: {svm_report['2']['precision']}, {svm_report['2']['recall']}, {svm_report['2']['f1-score']}")
print(f"SVM Class 3: {svm_report['3']['precision']}, {svm_report['3']['recall']}, {svm_report['3']['f1-score']}")
print(f"SVM Average: {svm_report['weighted avg']['precision']}, {svm_report['weighted avg']['recall']}, {svm_report['weighted avg']['f1-score']}")
print()
print(f"Regression Class 1: {lr_report['1']['precision']}, {lr_report['1']['recall']}, {lr_report['1']['f1-score']}")
print(f"Regression Class 2: {lr_report['2']['precision']}, {lr_report['2']['recall']}, {lr_report['2']['f1-score']}")
print(f"Regression Class 3: {lr_report['3']['precision']}, {lr_report['3']['recall']}, {lr_report['3']['f1-score']}")
print(f"Regression Average: {lr_report['weighted avg']['precision']}, {lr_report['weighted avg']['recall']}, {lr_report['weighted avg']['f1-score']}")
print()
print(f"Naive Bayes Class 1: {nb_report['1']['precision']}, {nb_report['1']['recall']}, {nb_report['1']['f1-score']}")
print(f"Naive Bayes Class 2: {nb_report['2']['precision']}, {nb_report['2']['recall']}, {nb_report['2']['f1-score']}")
print(f"Naive Bayes Class 3: {nb_report['3']['precision']}, {nb_report['3']['recall']}, {nb_report['3']['f1-score']}")
print(f"Naive Bayes Average: {nb_report['weighted avg']['precision']}, {nb_report['weighted avg']['recall']}, {nb_report['weighted avg']['f1-score']}")
