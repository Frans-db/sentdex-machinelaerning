from sklearn import preprocessing, model_selection, neighbors
import numpy as np
import pandas as pd

# load the dataset
df = pd.read_csv('breast-cancer-wisconsin.data')
# replace missing data
df.replace('?', -99999, inplace=True)
# remove id 
df.drop(['id'], 1, inplace=True)

# create features and labels
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
# creating train and test data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# train classifier
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
# test classifier
accuracy = clf.score(X_test, y_test)
print(accuracy)

# unknown data
example_measures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
example_measures = example_measures.reshape(1, -1)
# prediction
prediction = clf.predict(example_measures)
print(prediction)