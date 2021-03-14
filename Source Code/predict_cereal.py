# -*- coding: utf-8 -*-
"""
Aniket
SVR, Linear Regression and XGBOOST for Rating prediction

"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_tree
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import time
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import category_encoders as ce
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


data = pd.read_csv('/cereal.csv')
#from https://www.kaggle.com/fedesoriano/stroke-prediction-dataset

dataset = data.values
data.drop(['name'], axis = 1, inplace = True)
print(data)

ce_one_hot = ce.OneHotEncoder(cols= ['mfr', 'type'])
OH_data = ce_one_hot.fit_transform(data)
print(OH_data)

seed = 7
test_size = 0.33
Y = OH_data.rating
OH_data.drop(['rating'], axis = 1, inplace = True)

X_train, X_test, y_train, y_test = train_test_split(OH_data, Y.values, random_state = 0)


model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
score = mean_absolute_error(y_test, prediction)
print(model.score(X_test,y_test))

from sklearn.svm import SVR
model = SVR(C=100,epsilon=0.2)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
score = mean_absolute_error(y_test, prediction)
print(model.score(X_test,y_test))

model1 = XGBRegressor(objective='reg:squarederror',)
model1.fit(X_train, y_train)
prediction = model1.predict(X_test)
score = mean_absolute_error(y_test, prediction)
print(model1.score(X_test,y_test))

ce_one_hot = ce.OneHotEncoder(cols= ['mfr', 'type'])
OH_data = ce_one_hot.fit_transform(data)
OH_data_normalised = (OH_data - OH_data.min())/(OH_data.max()-OH_data.min())

seed = 7
test_size = 0.33
Y = OH_data.rating
OH_data_normalised.drop(['rating'], axis = 1, inplace = True)
X_train, X_test, y_train, y_test = train_test_split(OH_data_normalised, Y.values, random_state = 0)
model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)

score = mean_absolute_error(y_test, prediction)
print(score)

cdf = pd.DataFrame(model.coef_, X_train.columns, columns=['Coefficients'])

print(cdf)

cdf.plot.bar()

t = np.arange(prediction.size)
plt.plot(t, prediction, 'r')
plt.plot(t, y_test, 'b')
plt.show()

