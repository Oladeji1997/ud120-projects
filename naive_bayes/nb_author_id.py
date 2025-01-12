#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("/content/ud120-projects/tools")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()
clf = GaussianNB()

t0 = time()
# # < your clf.fit() line of code >
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
# # < your clf.predict() line of code >
pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

accuracy = accuracy_score(labels_test, pred)
print("Accuracy:", accuracy)
##############################################################
