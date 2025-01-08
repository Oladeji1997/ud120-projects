#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the terrain data
features_train, labels_train, features_test, labels_test = makeTerrainData()

# Create an SVM classifier with a linear kernel
clf = SVC(kernel="linear")

# Measure training time and fit the model
t0 = time.time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time.time() - t0, 3), "s")

# Measure prediction time and make predictions
t0 = time.time()
pred = clf.predict(features_test)
print("Predicting Time:", round(time.time() - t0, 3), "s")

# Calculate accuracy
acc = accuracy_score(labels_test, pred)
print("Accuracy:", acc)

# Visualize the decision boundary (optional)
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass

# Function to return accuracy for submission
def submitAccuracy():
    return acc


#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
