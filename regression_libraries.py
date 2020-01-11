from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection
from matplotlib import style
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import quandl
import math

# matplotlib styling
style.use('ggplot')

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
forecast_out = int(math.ceil(0.1 * len(df)))
print(f'Shifting data {forecast_out} days into the future')
# creating the label (data we want to predict)
df['label'] = df[forecast_col].shift(-forecast_out) 
# extracting the features
X = np.array(df.drop(['label'], 1))
# scaling the data 
X = preprocessing.scale(X)
# features to predict unknown data
X_lately = X[-forecast_out:]
# removing forecast_out elements
X = X[:-forecast_out]
# extracting the labels
df.dropna(inplace=True)
y = np.array(df['label'])
# creating train and test data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# create classifier and train it
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
# test classifier
accuracy = clf.score(X_test, y_test)
print(f'clf accuracy is {accuracy}')

# predicting unknown data
forecast_set = clf.predict(X_lately)

# creating data for plotting
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
day = 24 * 60 * 50 # day in seconds
next_unix = last_unix + day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()