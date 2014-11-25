# -*- coding: utf-8 -*-
"""
Created on Tue Oct 07 10:06:55 2014

@author: Kiran
"""

from __future__ import division

import re
from nltk.stem import PorterStemmer
import numpy as np
import scipy.io as scipyio
import sklearn.svm as svm

NUM_WORDS_IN_VOCAB_DICT = 1899

def get_vocab_dict():
  vocab_filename = 'vocab.txt'
  prog = re.compile(r'(\d+)\t(\w+)')
  with open(vocab_filename) as f:
    lines = f.readlines()
  match_groups = [prog.match(line) for line in lines]
  vocab_dict = {m.group(2): int(m.group(1))-1 for m in match_groups if m} # -1 massaging for numpy indexing
  vocab_inv_dict = {(int(m.group(1))-1): m.group(2) for m in match_groups if m}
  return vocab_dict, vocab_inv_dict

def read_file(filename):
  with open(filename, 'r') as f:
    file_contents = f.read()
  return file_contents

def process_email(email_contents):
  vocab_dict, __ = get_vocab_dict()
  email_contents = email_contents.lower()
  email_contents = re.sub(r'<[^<>]+>',' ', email_contents)
  email_contents = re.sub('[0-9]+', 'number', email_contents)
  email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
  email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
  email_contents = re.sub('[$]+', 'dollar', email_contents)
  strlist = re.split(r" |@|$|/|#|\.|-|:|&|\*|\+|=|[|]|\?|!|\(|\)|\{|\}|,|'|'|\"|>|_|<|;|%|\n|\r", email_contents)
  strlist = [re.sub('[^a-zA-Z0-9]','', x) for x in strlist]
  stemmer = PorterStemmer()
  strlist = [stemmer.stem(x) for x in strlist]
  word_indices = [vocab_dict[x] for x in strlist if vocab_dict.has_key(x)]
  return word_indices

def get_email_features(word_indices):
  x = np.zeros((NUM_WORDS_IN_VOCAB_DICT), dtype=np.float64)
  x[[word_indices]] = 1.0
  return x

def exercise_part1():
  email_contents = read_file('emailSample1.txt')
  print(email_contents)
  word_indices = process_email(email_contents)
  print(word_indices)
  email_features = get_email_features(word_indices)
  print(email_features)

def loadSpamTrainAndTest():
  data_train = scipyio.loadmat('spamTrain.mat')
  X = data_train['X']
  y = data_train['y'].reshape((-1))
  data_test = scipyio.loadmat('spamTest.mat')
  print(data_test.keys())
  X_test = data_test['Xtest']
  y_test = data_test['ytest'].reshape((-1))
  return X, y, X_test, y_test

def trainLinearSVM(X, y, C):
  clf = svm.LinearSVC(C=C)
  clf.fit(X, y)
  return clf

def exercise_part2():
  X, y, X_test, y_test = loadSpamTrainAndTest()
  clf = trainLinearSVM(X, y, 0.1)
  y_pred_train = clf.predict(X)
  print('training accuracy: ' + 
        str(np.count_nonzero(y_pred_train == y)/np.size(y_pred_train)) )
  y_pred_test = clf.predict(X_test)
  print('test accuracy: ' + 
        str(np.count_nonzero(y_pred_test == y_test)/np.size(y_pred_test)) )
  clf_weights = (clf.coef_).reshape((-1)) # the theta params for the feature linear separator
  sorted_idx = np.argsort(clf_weights)[::-1]
  __, vocab_inv_dict = get_vocab_dict()
  for idx, top_idx in enumerate(sorted_idx[0:15]):
    print('Predictor: #' + str(idx+1) + ' ' + vocab_inv_dict[top_idx]+ ', weight: {:2.2f}'.format(clf_weights[top_idx]))

def try_sample_email(email_filename, clf):
  email_contents = read_file(email_filename)
  print(email_contents)
  word_indices = process_email(email_contents)
  email_features = get_email_features(word_indices)
  if clf.predict(email_features)[0]==1:
    print(email_filename + ' appears to be spam')
  else:
    print(email_filename + ' appears to be non-spam')    
    

def exercise_part3():
  X, y, __, __ = loadSpamTrainAndTest()
  clf = trainLinearSVM(X, y, 0.1)
  try_sample_email('spamSample1.txt', clf)
  try_sample_email('spamSample2.txt', clf)
  try_sample_email('emailSample1.txt', clf)
  


def process_main():
  exercise_part1()
  exercise_part2()
  exercise_part3()
  

if __name__ == '__main__':
  process_main()