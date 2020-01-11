import pandas as pd
import quandl

# get the dataset
df = quandl.get('WIKI/GOOGL')
# extract useful labels
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
# percentage change between low and high
df['HL_percent'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100
# percentage change between open and close
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
# extract features
df = df[['Adj. Close', 'HL_percent', 'PCT_change', 'Adj. Volume']]

print(df.head())