import matplotlib as mp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv("unclean-wine-quality.csv")
dataset = dataset.iloc[:, 1:-1]
NaN = np.where(dataset.isnull())
NaNNum = dataset.isnull().sum()
Dash = np.where(dataset == '-')
DashNum = (dataset == '-').sum()
print(NaN)
print(NaNNum.sum())

print(Dash)
print(DashNum.sum())

TotalEmpty = NaNNum.sum()+DashNum.sum()
print('\n\n\n\n\n\n')
print(TotalEmpty)

dataset.mask(dataset == '-', other=np.nan, inplace=True)
dataset = dataset.astype('float64')

# Q2
cols = {
    'fixed acidity': 0,
    'volatile acidity': 0,
    'citric acid': 0,
    'residual sugar': 0,
    'chlorides': 1,
    'free sulfur dioxide': 0,
    'total sulfur dioxide': 0,
    'density': 0,
    'pH': 1,
    'sulphates': 1,
    'alcohol': 0
}

for col, val in cols.items():
    dataset[col].fillna(val, inplace=True)

nan_count_after = dataset.isnull().sum().sum()
print(nan_count_after)

# Q3
dataset = pd.read_csv("unclean-wine-quality.csv")
dataset = dataset.iloc[:, 1:-1]
dataset.mask(dataset == '-', other=np.nan, inplace=True)
dataset = dataset.astype('float64')

dataset.fillna(method='ffill', inplace=True)

print("Value at index [16, 0]:", dataset.iloc[16, 0])
print("Value at index [17, 0]:", dataset.iloc[17, 0])

# Q4
dataset = pd.read_csv("unclean-wine-quality.csv")
dataset = dataset.iloc[:, 1:-1]
dataset.mask(dataset == '-', other=np.nan, inplace=True)
dataset = dataset.astype('float64')

dataset = dataset.interpolate(method='linear')

print("Value at index [16, 0]:", dataset.iloc[16, 0])
print("Value at index [17, 0]:", dataset.iloc[17, 0])

# Q5
dataset = pd.read_csv("noisy-sine.csv")
fig, ax = plt.subplots(figsize=(10,10))
n_sample = 700
x_in = np.arange(n_sample)
ax.plot(x_in,dataset, linewidth=1)

dataset = pd.read_csv("noisy-sine.csv")
win5 = dataset.rolling(5).mean()
ax.plot(x_in, win5, linewidth=1)

dataset = pd.read_csv("noisy-sine.csv")
win31 = dataset.rolling(31).mean()
ax.plot(x_in, win31, linewidth=1)

dataset = pd.read_csv("noisy-sine.csv")
win51 = dataset.rolling(51).mean()
ax.plot(x_in, win51, linewidth=1)

ax.legend(['Noisy','5','31','51'])
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# Q6
dataset = pd.read_csv("ECG-sample.csv")
x_in = range(len(dataset))

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(x_in, dataset)

features = pd.DataFrame(columns=['mean','std','max','min'])
window_size = 31
features['mean'] = dataset.rolling(window=window_size).mean()
features['std'] = dataset.rolling(window=window_size).std()
features['max'] = dataset.rolling(window=window_size).max()
features['min'] = dataset.rolling(window=window_size).min()

print(features)

std = features.iloc[:, 1]
x_in = range(len(std))

fig,ax = plt.subplots(figsize=(10,10))
ax.plot(x_in, std)
ax.set_xlabel('Number of the window')
ax.set_ylabel('value of the std')
plt.show()