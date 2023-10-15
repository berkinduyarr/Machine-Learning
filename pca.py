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

veriler = pd.read_csv("Wine.csv")
#pd.read_csv("veriler.csv")


X = veriler.iloc[:, 0:13].values #bağımsız değişkenler
y = veriler.iloc[:, 13].values #bağımlı değişken
 


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

#pca dönüşümünden önce gelen LR
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#pca dönüşümünden sonra gelen LR
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)

#tahminler
y_pred = classifier.predict(X_test)
y_pred2  = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
#actual / PCA olmadan çıkan sonuç
print('gercek / PCAsiz')
cm = confusion_matrix(y_test, y_pred)
print(cm)

#actual / PCA çıkan sonuç
print('gercek / PCA ile')
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

#PCA sonrası / PCA öncesi
print('PCAsiz ve PCAli')
cm3 = confusion_matrix(y_pred, y_pred2)
print(cm3)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)

X_train_lda = lda.fit_transform(X_train,y_train)
X_test_lda = lda.transform(X_test)

#LDA dönüşümünden sonra

classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda,y_train)

#LDA verisini tahmin et
y_pred_lda = classifier_lda.predict(X_test_lda)

#LDA Sonrası / orijinal
print('lda ve orijinal')
cm4 = confusion_matrix(y_pred, y_pred_lda)
print(cm4)                                