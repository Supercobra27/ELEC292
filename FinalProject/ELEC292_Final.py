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

if os.path.exists("main_data.hdf5"):
    os.remove("main_data.hdf5")

m1 = process_data('member1', 'RyanWalk.csv', 'RyanJump.csv')
m2 = process_data('member2', 'FosterWalk.csv', 'FosterJump.csv')
m3 = process_data('member3', 'JuliaWalk.csv', 'JuliaJump.csv')
segment(get_data('member2')[0], 'walk', 'member2') #the worst
segment(get_data('member2')[1], 'jump', 'member2')

def plot_data(data, activity, name, window_size):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(data[1], data[2].rolling(window_size).mean(),
                 label="Linear Acceleration x (m/s^2)")
    axes[0].set_ylabel("X Acceleration (m/s^2)")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(data[1], data[3].rolling(window_size).mean(),
                 label="Linear Acceleration y (m/s^2)", color='orange')
    axes[1].set_ylabel("Y Acceleration (m/s^2)")
    axes[1].legend()
    axes[1].grid(True)
    axes[2].plot(data[1], data[4].rolling(window_size).mean(),
                 label="Linear Acceleration z (m/s^2)", color='green')
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Z Acceleration (m/s^2)")
    axes[2].legend()
    axes[2].grid(True)
    plt.suptitle("Acceleration VS Time for "+name+" | "+activity, fontsize=20)

def visualize_data(member_arr):
    for member in member_arr:
        walk_data, jump_data = get_data(member)
        plot_data(walk_data, 'walking', member, 5)
        plot_data(jump_data, 'jumping', member, 5)

member_arr = ['member1','member2', 'member3']
plot_data(get_segment('walk1004', 'member2'), 'walking', 'member', 5)
plt.show()
# ADD MORE GRAPHS

def extract_from(data, window_size):
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
        return features


def extract_features(member_arr, window_size):
    for member in member_arr:
        walk_data, jump_data = get_data(member)
        walk_data = walk_data.iloc[:, 3:]
        jump_data = jump_data.iloc[:, 3:]
        features = [extract_from(walk_data, window_size), extract_from(jump_data, window_size)]
        return features

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
