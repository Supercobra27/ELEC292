# Author(s): Ryan Silverberg
import numpy as np
import h5py
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from hdffunctions import process_data, segment, LabelData, get_data, get_segment
from plotfeatures import plot_data, visualize_data, extract_from, extract_features

if os.path.exists("main_data.hdf5"):
    os.remove("main_data.hdf5")

m1 = process_data('member1', 'RyanWalk.csv', 'RyanJump.csv')
m2 = process_data('member2', 'FosterWalk.csv', 'FosterJump.csv')
m3 = process_data('member3', 'JuliaWalk.csv', 'JuliaJump.csv')
segment(get_data('member2')[0], 'walk', 'member2') #the worst
segment(get_data('member2')[1], 'jump', 'member2')

member_arr = ['member1','member2', 'member3']
plot_data(get_segment('walk1004', 'member2'), 'walking', 'member', 5)
plt.show()
# ADD MORE GRAPHS

data = get_segment('walk1004', 'member2')
data2 = data
data = data.iloc[:, 2:5]
print(data)

window_size = 5
features = pd.DataFrame()
for column in data.columns:
    feature_column = pd.DataFrame()
    feature_column['mean'] = data[column].rolling(window=window_size).mean()
    feature_column['std'] = data[column].rolling(window=window_size).std()
    feature_column['max'] = data[column].rolling(window=window_size).max()
    feature_column['min'] = data[column].rolling(window=window_size).min()
    feature_column['kurtosis'] = data[column].rolling(window=window_size).kurt()
    feature_column['skew'] = data[column].rolling(window=window_size).skew()
    feature_column['median'] = data[column].rolling(window=window_size).median()
    feature_column['range'] = data[column].rolling(window=window_size).max() - data[column].rolling(window=window_size).min()
    feature_column['sum'] = data[column].rolling(window=window_size).sum()
    feature_column['variance'] = data[column].rolling(window=window_size).var()
    features = pd.concat([features, feature_column], axis=1).dropna()

# features = extract_from(data, window_size)

print(features)
# Normalization
genscale = StandardScaler()
data = pd.DataFrame(genscale.fit_transform(data))
print(data)

scaler = StandardScaler()
model = LogisticRegression()

data = data2.iloc[:, 2:5]
labels = data2.iloc[:, 0]

# Split the data into training and testing sets (90% for training, 10% for testing)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

clf = make_pipeline(scaler, model)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
