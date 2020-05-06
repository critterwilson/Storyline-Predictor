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
colnames = ["w_king_col", "w_king_row", "w_rook_col", "w_rook_row", 
            "b_king_col", "b_king_row", "depth_of_win"]

data = pd.read_csv("krkopt.data", header = None, skipinitialspace = False, 
                   names = colnames, na_values=["?"])

# Data Pre-Processing (start w/ 1 on "..._col" to stay consistent with "..._row")
data["w_king_col"] = data.w_king_col.map({"a": 1, "b": 2, "c":3, "d":4})

data["w_rook_col"] = data.w_rook_col.map({"a": 1, "b": 2, "c":3, "d":4,
                                          "e": 5, "f": 6, "g":7, "h":8})

data["b_king_col"] = data.b_king_col.map({"a": 1, "b": 2, "c":3, "d":4,
                                          "e": 5, "f": 6, "g":7, "h":8})

data["depth_of_win"] = data.depth_of_win.map({"zero": 0, "one": 1, "two":2, 
                                          "three":3, "four": 4, "five": 5, 
                                          "six": 6, "seven": 7, "eight": 8, 
                                          "nine": 9, "ten": 10, "eleven": 11,
                                          "twelve": 12, "thirteen": 13,
                                          "fourteen": 14, "fifteen": 15, 
                                          "sixteen": 16, "draw": -1})


x = data.drop(columns=["depth_of_win"]).to_numpy()
y = data["depth_of_win"].to_numpy()

# Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Create Decision Tree
id3 = tree.DecisionTreeClassifier()
id3.fit(x_train, y_train)
y_pred = id3.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("id3 Accuracy: {}".format(accuracy))

gnb = GaussianNB()
gnb.fit(x_train, y_train )
y_pred = gnb.predict(X = x_test)
accuracy = accuracy_score(y_test, y_pred)
print("GNB Accuracy: {}".format(accuracy))

knn = KNeighborsClassifier(n_neighbors=18)
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

forest = RandomForestClassifier(n_estimators=500)
forest.fit(x_train, y_train)
y_pred = forest.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Forest Accuracy: {}".format(accuracy))
