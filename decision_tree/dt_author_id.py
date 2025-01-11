#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from sklearn.tree import DecisionTreeClassifier
sys.path.append("/content/ud120-projects/tools")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from email_preprocess import preprocess

features_train, features_test, labels_train, labels_test = preprocess()

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(min_samples_split=40, random_state=42)
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time() - t0, 3), "s")

# Measure prediction time
t1 = time()
pred = clf.predict(features_test)
print("Prediction Time:", round(time() - t1, 3), "s")

# Compute accuracy
accuracy = accuracy_score(labels_test, pred)
print("Decision Tree Accuracy:", round(accuracy, 3))


