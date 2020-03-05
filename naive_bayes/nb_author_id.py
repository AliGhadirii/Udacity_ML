#!/usr/bin/python3

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
cls = GaussianNB()
t0 = time()
cls.fit(features_train,labels_train)
print ("training time:", round(time()-t0, 3), "s")
t0 = time()
pred = cls.predict(features_test)
print ("predict time:", round(time()-t0, 3), "s")
print(accuracy_score(pred,labels_test))



#########################################################
### your code goes here ###


#########################################################


