#!/usr/bin/python3


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

### it's all yours from here forward!



import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score,recall_score
from sklearn.model_selection import train_test_split
import numpy as np



features_train, features_test, labels_train, labels_test = train_test_split(features,labels,random_state=42,test_size=0.3)
clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
print(labels_test.count(1))
print(len(labels_test))
print(np.where(pred == 1))
print(np.where(np.asarray(labels_test == 1)))
print(accuracy_score(pred,labels_test))
print(precision_score(labels_test,pred))
print(recall_score(labels_test,pred))



### your code goes here 


