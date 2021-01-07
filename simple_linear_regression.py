# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 21:19:03 2021

@author: Ken
"""
#simple linear  regression

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv("Salaries.csv")
X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1].values

#split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 1/3, random_state=0)

#fitting simple linear regression to train test
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#make it a 2-dimensional array
X_train = X_train.reshape(-1,1)
Y_train = Y_train.reshape(-1,1)
regressor.fit(X_train, Y_train)

#predicting test set results
#reshape our test set too(2-dimension)
X_test = X_test.reshape(-1,1)
#using X_test to predict Y_test values
Y_pred = regressor.predict(X_test)

#visualising results: training set
plt.scatter(X_train, Y_train)
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('Salary vs Experience (Training set results)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualising results: test set
plt.scatter(X_test, Y_test)
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('Salary vs Experience (Test set results)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
















