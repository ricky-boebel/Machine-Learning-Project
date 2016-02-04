#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL",0)

#other outliers
key_list = [k for k in data_dict.keys() if data_dict[k]["salary"] != 'NaN' and data_dict[k]["salary"] > 1000000 and data_dict[k]["bonus"] > 5000000]



# We will  leave these in as they are valid datapoints for investigation

### Task 3: Create new feature(s)
for person in data_dict.values():
    person['poi_ratio']=0
    person['total_messages']=0
    if float(person['from_messages'])>0 or float(person['to_messages'])>0:
        person['total_messages']=(float(person['to_messages'])+float(person['from_messages']))
        person['poi_ratio']=(float(person['from_this_person_to_poi'])+float(person['from_poi_to_this_person']))/person['total_messages']

features_list = ['poi','deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees','to_messages',  'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

#feature combinations found by SelectKBest
f2 =['poi','loan_advances', 'deferred_income']
f4= ['poi', 'loan_advances', 'deferred_income', 'expenses', 'from_this_person_to_poi' ]
f6 = ['poi', 'deferral_payments', 'loan_advances', 'restricted_stock_deferred', 'deferred_income', 'expenses', 'from_this_person_to_poi']
f8=['poi', 'deferral_payments', 'loan_advances', 'restricted_stock_deferred', 'deferred_income', 'expenses', 'other', 'long_term_incentive', 'from_this_person_to_poi']

features_list.extend(['total_messages','poi_ratio'])

features_list=f4

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#find  features that matter, how the scores were found from SelectKBest

selection= SelectKBest(k='all')
selection.fit(features_train,labels_train)
feature_names=selection.get_support(indices=False)
scores=selection.scores_
ctr=0

f1=[]

for a in feature_names:
    if a==True:
        f1.append(features_list[ctr])
        f1.append(scores[ctr])
        ctr+=1
    else:
        print features_list[ctr]
        ctr+=1
print "features and scores:"
print f1


clf = tree.DecisionTreeClassifier(min_samples_split=7, splitter='best', criterion="entropy" )

clf.fit(features_train, labels_train)

pred=clf.predict(features_test)

acc= accuracy_score(pred, labels_test)
print "Performance Metrics:"
print acc
print recall_score( labels_test,pred)
print precision_score( labels_test, pred)



dump_classifier_and_data(clf, my_dataset, features_list)
