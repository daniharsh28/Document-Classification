__author__ = 'Harsh'
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import csv
# Load column names
columns = []
str_word_indices = raw_input('Enter word indices path command')
reader = csv.reader(open(str_word_indices,'rb'))

for row in reader:
    columns.append(row[0])
# Load training features
str_train = raw_input('Enter path for training.csv')
features = pd.read_csv(str_train,names=columns)
#Load training labels
str_labels = raw_input('Enter path for training labels')
str_test_features = raw_input('Enter path for test.csv')
actual_test_str = raw_input('Enter path for test_labels.csv')
training_labels = pd.read_csv(str_labels,header=None)
training_labels = np.array(training_labels[0])
training_labels = training_labels.transpose()
print training_labels
#Train the Naive Bayes Classifier
print 'Training Naive Bayes Classifier'
mnb = MultinomialNB(alpha=0.5,fit_prior=True).fit(features,training_labels)

test_features = pd.read_csv(str_test_features,names=columns)
print 'Now predict for test data'
predict_label_test = mnb.predict(test_features)
#Load actual test data
actual_test_data = pd.read_csv(actual_test_str,header=None)
print accuracy_score(actual_test_data,predict_label_test)
