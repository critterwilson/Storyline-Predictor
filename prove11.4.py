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
colnamesCars = ["buying", "maint", "doors", "persons", "lug_boot", "safety", 
                "car_class"]
dataCars = pd.read_csv("car.data", header = None, skipinitialspace = False, 
                   names = colnamesCars, na_values=["?"])
# convert buying to categories
dataCars.buying = dataCars.buying.astype('category')
dataCars["buying_cat"] = dataCars.buying.map({"vhigh": 4, "high": 3, "med": 2,
        "low": 1})
# convert maint to categories
dataCars.maint = dataCars.maint.astype('category')
dataCars["maint_cat"] = dataCars.maint.map({"vhigh": 4, "high": 3, "med": 2,
        "low": 1})
# convert lug_boot to categories
dataCars.lug_boot = dataCars.lug_boot.astype('category')
dataCars["lug_boot_cat"] = dataCars.lug_boot.map({"big": 3, "med": 2, 
        "small": 1})
# convert safety to categories
dataCars.safety = dataCars.safety.astype('category')
dataCars["safety_cat"] = dataCars.safety.map({"high": 3, "med": 2, "low": 1})
# convert class to categories
dataCars.car_class = dataCars.car_class.astype('category')
dataCars["car_class_cat"] = dataCars.car_class.map({"vgood": 4, "good": 3, 
        "acc": 2, "unacc": 1})
# replace "5more" in doors
dataCars["doors"] = dataCars.doors.replace({"5more": 5})
# replace "more" in persons
dataCars["persons"] = dataCars.persons.replace({"more": 6})
# drop unnecessary columns
dataCars = dataCars.drop(columns=["buying", "maint", "lug_boot", "safety", 
                                  "car_class"])
# convert the data to numpy arrays, because that's what sk-learn likes
x = dataCars.drop(columns=["car_class_cat"]).to_numpy()
y = dataCars["car_class_cat"].to_numpy()

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

knn = KNeighborsClassifier(n_neighbors=4)
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
