# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 13:20:56 2023

@author: Berkin
"""

#ders 6 : kutuphanelerin yuklenmesi

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# kod bolumu
# veri yukleme

veriler = pd.read_csv("veriler.csv")
#pd.read_csv("veriler.csv")

print(veriler)

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)
    


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# buradan itibaren sınıflandırma algoritması
#1. logistic regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train) # egitim

y_pred = logr.predict(X_test) #tahmin
print(y_pred)
print(y_test)

#karmasiklik matrisi
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 2. KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

# 3. SVC (SVM classifier)
from sklearn.svm import SVC
svc = SVC(kernel='poly')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('SVC')
print(cm)

