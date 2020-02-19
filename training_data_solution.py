#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:45:11 2020

@author: imran
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('train.csv')
y = dataset.iloc[:, 0]
X = dataset.drop(['label'], axis=1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=2)


import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()

classifier.add(Dense(output_dim=392, init='uniform', activation='relu', input_dim=784))

#classifier.add(Dense(output_dim=784, init='uniform', activation='relu'))

#classifier.add(Dense(output_dim=1568, init='uniform', activation='relu'))

classifier.add(Dense(output_dim=10, init='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=100, epochs=50)

y_pred = classifier.predict(X_test, batch_size=100, verbose=1)


y_pred_df = pd.DataFrame(data=y_pred[0:,0:]) 

y_pred_trans = pd.get_dummies(y_pred_df).idxmax(1)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_trans)

