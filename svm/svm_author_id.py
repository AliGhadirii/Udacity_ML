#!/usr/bin/python3

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
#features_train = features_train[:len(features_train)//100]
#labels_train = labels_train[:len(labels_train)//100]
### make sure you use // when dividing for integer division
cls = SVC(kernel='rbf', C = 10000.0)
t0 = time()
cls.fit(features_train,labels_train)
print ("training time:", round(time()-t0, 3), "s")
t0 = time()
pred = cls.predict(features_test)
print ("predict time:", round(time()-t0, 3), "s")
print(accuracy_score(pred,labels_test))
print(pred[10],pred[26],pred[50])
print(np.count_nonzero(pred == 1))


#########################################################
### your code goes here ###

#########################################################


