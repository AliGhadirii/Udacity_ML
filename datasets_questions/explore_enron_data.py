#!/usr/bin/python3

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

ans = list(enron_data.values())
temp = 0
for i in ans:
    if i['poi']:
       temp += 1

temp2 = 0
for j in ans:
    if j['salary'] != 'NaN':
        temp2 += 1

temp3 = 0
for j in ans:
    if j['email_address'] != 'NaN':
        temp3 += 1

temp4 = 0
for j in ans:
    if j['total_payments'] == 'NaN' and j['poi']:
        temp4 += 1

print(temp,'  ',temp2,'  ',temp3, ' ', temp4/len(enron_data))

print(enron_data['COLWELL WESLEY']['from_this_person_to_poi'])

print(enron_data['SKILLING JEFFREY K']['total_payments'])
print(enron_data['FASTOW ANDREW S']['total_payments'])
print(enron_data['LAY KENNETH L']['total_payments'])

print(enron_data.values())



