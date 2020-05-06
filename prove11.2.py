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
# Load in the Data
data = pd.read_csv("game_teams_stats.data", header=0, skipinitialspace = False, 
                   na_values=["?"])
# PreProcess the Data
data.HoA = data.HoA.map({"away":0, "home":1})
data.won = data.won.replace({False:0, True:1})
data.settled_in = data.settled_in.map({"REG":0, "OT":1, "SO":2})
data.head_coach = data.head_coach.astype('category').cat.codes

# Prepare data for Keras
x = data.drop(columns=["won", "game_id", "team_id", "head_coach"])
y = data.won

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
