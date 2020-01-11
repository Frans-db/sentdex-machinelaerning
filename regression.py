import pandas as pd
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
# creating the label (data we want to predict)
df['label'] = df[forecast_col].shift(-forecast_out) 
df.dropna(inplace=True)

print(df.head())