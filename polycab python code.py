# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 16:39:02 2026

@author: savin
"""

import pandas as pd
import numpy as np

df = pd.read_excel("C:/Users/savin/OneDrive/Desktop/Ratul sir's/polycab credit card conversion/new polycab/for_python.xlsx")

# Putting feature variable to X
X = df.drop('y',axis=1)
# Putting response variable to y
y = df['y']


# now lets split the data into train and test
from sklearn.model_selection import train_test_split

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
X_train.shape, X_test.shape


#Random Forest ------------------------------------------
from sklearn.ensemble import RandomForestClassifier

classifier_rf = RandomForestClassifier(random_state=42, max_depth=5,
                                       n_estimators=100)


classifier_rf.fit(X_train, y_train)



train_proba_rf = classifier_rf.predict_proba(X_train)
test_proba_rf = classifier_rf.predict_proba(X_test)

#----------------------------------------------------------

#Logistic regression --------------------------------------

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

train_proba_logistic = model.predict_proba(X_train) 
test_proba_logistic = model.predict_proba(X_test)

#-----------------------------------------------------------

#XGBoost----------------------------------------------------

from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

train_proba_xgboost = model.predict_proba(X_train)
test_proba_xgboost = model.predict_proba(X_test)
#----------------------------------------------------------

#svm(supprot vector machine)------------------------------------------------------

from sklearn.svm import SVC

svm_model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

svm_model.fit(X_train, y_train)

train_proba_svm = svm_model.predict_proba(X_train)
test_proba_svm = svm_model.predict_proba(X_test)


#---------------------------------------------------------------------------

#mlp classifier (Multi-Layer Perceptron)

from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier(
    random_state=42,
    max_iter=500
)

mlp_model.fit(X_train, y_train)

train_proba_mlp = mlp_model.predict_proba(X_train)
test_proba_mlp = mlp_model.predict_proba(X_test)

#----------------------------------------------------

#Naive Bayes

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)

train_proba_naive_bayes = model.predict_proba(X_train)
test_proba_naive_bayes = model.predict_proba(X_test)

#----------------------------------------------------

#KNN classifier

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(
    n_neighbors=5
)

knn_model.fit(X_train, y_train)

train_proba_knn = knn_model.predict_proba(X_train)
test_proba_knn = knn_model.predict_proba(X_test)




#------------------------------------------------------






