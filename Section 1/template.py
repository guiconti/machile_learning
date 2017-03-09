#   Importing libraries
import numpy as numPy
import matplotlib.pyplot as matPlot
import pandas as panda

#   Importing dataset
dataset = panda.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

#   Splitting the dataset into the training set and the test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#   Feature scaling
"""from sklearn.preprocessing import StandardScaler
scaleX = StandardScaler()
X_train = scaleX.fit_transform(X_train)
X_test = scaleX.transform(X_test)"""