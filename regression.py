from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection
import pandas as pd
import numpy as np
import quandl
import math


# get the dataset
df = quandl.get('WIKI/GOOGL')
# extract useful labels
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
# percentage change between low and high
df['HL_percent'] = ((df['Adj. High'] - df['Adj. Low']) / df['Adj. Low']) * 100
# percentage change between open and close
df['PCT_change'] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']) * 100
# extract features
df = df[['Adj. Close', 'HL_percent', 'PCT_change', 'Adj. Volume']]

# column we want to forceast
forecast_col = 'Adj. Close'
# fill non existant data with -99999
df.fillna(-99999, inplace=True)
# how many days into the future we want to predict
forecast_out = int(math.ceil(0.01 * len(df)))
print(f'Shifting data {forecast_out} days into the future')
# creating the label (data we want to predict)
df['label'] = df[forecast_col].shift(-forecast_out) 
df.dropna(inplace=True)

# extracting the features and labels
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
# scaling the data 
X = preprocessing.scale(X)
# creating train and test data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# create classifier and train it
clf = LinearRegression()
clf.fit(X_train, y_train)
# test classifier
accuracy = clf.score(X_test, y_test)
print(f'clf accuracy is {accuracy}')