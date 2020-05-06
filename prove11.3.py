# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# =============================================================================
# CHESS DATA SET
# =============================================================================
# Read in Data
colnames = ["party", "handicapped_infant", "water_cost", "budget_res", 
            "physician_fee_freeze", "el_salvador", "religion_school", 
            "satellite_ban", "nicaragua_aid", "missle", "immigration", 
            "fuel_corp_cut", "education_spend", "right_to_sue", "crime",
            "duty_free", "export_sa"]
data = pd.read_csv("house-votes-84.data", header = None, skipinitialspace = False, 
                   names = colnames, na_values=["?"])

data["party"] = data.party.map({"republican": 1, "democrat": 0})
# Replace all NaN with 2 (choosing not to vote, abstention)
data.fillna("a", inplace=True)

# targets, vs "classes"
x = pd.get_dummies(data.drop(columns=["party"])).to_numpy()
y = data.party.to_numpy()

# Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Create Decision Tree
id3 = tree.DecisionTreeClassifier()
id3.fit(x_train, y_train)
y_pred = id3.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("id3 Accuracy: {}".format(accuracy))

gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("GNB Accuracy: {}".format(accuracy))

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("KNN Accuracy: {}".format(accuracy))

############
# ADABOOST
############
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=100)
ada.fit(x_train, y_train)
y_pred = ada.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Ada Accuracy: {}".format(accuracy))

#################
# RANDOM FOREST
#################
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100)
forest.fit(x_train, y_train)
y_pred = forest.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Forest Accuracy: {}".format(accuracy))
