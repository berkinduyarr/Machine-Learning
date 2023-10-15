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

veriler = pd.read_csv("Churn_Modelling.csv")
#pd.read_csv("veriler.csv")

print(veriler)

#veri on isleme
X = veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values
    
#eksik veriler
#sci - kit learn


#encoder: Kategorik -> Numeric
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,1] = le2.fit_transform(X[:,2])


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],remainder="passthrough")
X = ohe.fit_transform(X)
X = X[:,1:]


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=0)

#verilerin olceklenmesi

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#yapay sinir ağları

import keras

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(6, init = "uniform", activation = "relu",input_dim =11))

classifier.add(Dense(6, init = "uniform", activation = "relu"))