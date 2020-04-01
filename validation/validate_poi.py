#!/usr/bin/python3


"""
    starter code for the validation mini-project
    the first step toward building your POI identifier!

    start by loading/formatting the data

    after that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list,sort_keys = '../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features,labels,random_state=42,test_size=0.3)
clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
print(accuracy_score(pred,labels_test))

### it's all yours from here forward!  


