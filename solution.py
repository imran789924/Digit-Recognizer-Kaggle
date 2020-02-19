#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:44:41 2020

@author: imran
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')
y = dataset.iloc[:, 0]
X = dataset.drop(['label'], axis=1)


#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=2)


import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()

classifier.add(Dense(output_dim=392, init='uniform', activation='relu', input_dim=784))

classifier.add(Dense(output_dim=392, init='uniform', activation='relu'))


classifier.add(Dense(output_dim=10, init='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

classifier.fit(X, y, batch_size=100, epochs=170)

y_pred = classifier.predict(X_test, batch_size=100, verbose=1)


y_pred_df = pd.DataFrame(data=y_pred[0:,0:]) 

y_pred_trans = pd.get_dummies(y_pred_df).idxmax(1)

results = pd.Series(y_pred_trans,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("my_submission.csv",index=False)




'''


classifier.add(Dense(output_dim=784, init='uniform', activation='relu'))
y_pred_df = pd.DataFrame(data=y_pred[1:,1:],    # values
              index=y_pred[1:,0],    # 1st column as index
              columns=y_pred[0,1:])  # 1st row as the column names



y_pred_df = pd.DataFrame(data=y_pred[1:,1:],    # values
              index=y_pred[1:,0],    # 1st column as index
              columns=y_pred[0,1:])  # 1st row as the column names



y_pred_df.columns = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

Cov = pd.read_csv("path/to/file.txt", 
                  sep='\t', 
                  names=["Sequence", "Start", "End", "Coverage"])

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_trans)
'''
