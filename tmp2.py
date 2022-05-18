import pandas as pd

# dataset2 = pd.read_excel("beijing.xlsx")
# dataset =pd.DataFrame(dataset2)
# dataset['date'] = pd.to_datetime(dataset['date'])
# dataset['month'] = dataset.date.dt.month
# df1 = dataset.groupby(['month'])["precipitation"].mean()
# print(dataset.head(10))
# print(dataset2.head(10))
# print(df1)

from sklearn import preprocessing
import numpy as np
X = np.array([[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]])
scaler= preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(X)
X_scaled = scaler.transform(X)
print(X)
print(X_scaled)
X1=scaler.inverse_transform(X_scaled)
print(X1)
print(X1[0, -1])
Y = X[:, 0].reshape(-1, 1)
scaler= preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(Y)
X_scaled = scaler.transform(Y)
print(X_scaled)
X1=scaler.inverse_transform(X_scaled)
print(X1)

X[:,0] = X1[:, 0]
print("--------------------")
print(X[:,0] )
print(X)