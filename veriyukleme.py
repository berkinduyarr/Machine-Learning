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

veriler = pd.read_csv("eksikveriler.csv")
#pd.read_csv("veriler.csv")

print(veriler)

#veri on isleme
boy = veriler[["boy"]]
print(boy)

boykilo = veriler[["boy","kilo"]]
print(boykilo)

x = 10

class insan:
    boy = 180 
    def kosmak(self,b):
        return b + 10
    # y = f(x)
    # f(x) = x+10
    
ali = insan()
print(ali.boy)
print(ali.kosmak(90))

l = [1,3,4] #liste

#eksik veriler 

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)
