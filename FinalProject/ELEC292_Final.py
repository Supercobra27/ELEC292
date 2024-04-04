# Author(s): Ryan Silverberg
import numpy as np
import h5py
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Data Processing
def process_data(member_name, walk_csv, jump_csv):
    walk_data = pd.read_csv(walk_csv)
    jump_data = pd.read_csv(jump_csv)
    with h5py.File('main_data.hdf5', 'a') as hdf:
        mg = hdf.create_group('/'+member_name)
        mg.create_dataset('walk', data=walk_data)
        mg.create_dataset('jump', data=jump_data)
        return walk_data, jump_data

m1walk, m1jump = process_data('member1', 'RyanWalk.csv', 'RyanJump.csv')
m2walk, m2jump = process_data('member2', 'FosterWalk.csv', 'FosterJump.csv')
m3walk, m3jump = process_data('member3', 'JuliaWalk.csv', 'JuliaJump.csv')

# Split the data into training and testing sets (90% for training, 10% for testing)
X_train, X_test, y_train, y_test = train_test_split(m1walk.iloc[:, :-1], m1walk.iloc[:, -1], test_size=0.1, random_state=42)


# Store the new dataset in the HDF5 file
with h5py.File('main_data.hdf5', 'a') as hdf:

    g1 = hdf.create_group('/model')
    g1.create_dataset('X_train', data=X_train)
    g1.create_dataset('X_test', data=X_test)
    g1.create_dataset('Y_train', data=y_train)
    g1.create_dataset('Y_test', data=y_test)

def get_data(dataset_name):
    with h5py.File('main_data.hdf5', 'r') as hdf:
        walk_data = pd.DataFrame(np.array(hdf['/'+dataset_name+'/walk']))
        jump_data = pd.DataFrame(np.array(hdf['/'+dataset_name+'/jump']))
        walk_time = walk_data.iloc[:, 0]
        jump_time = jump_data.iloc[:, 0]
        return walk_data, jump_data, walk_time, jump_time
    
def visualize_data(walk, jump, w_time, j_time, window_size):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    for i, ax in enumerate(axes):
        ax.plot(w_time, walk.iloc[:, i+1].rolling(window_size).mean(), linewidth=1, label='Walk')
        ax.plot(j_time, jump.iloc[:, i+1].rolling(window_size).mean(), linewidth=1, label='Jump')
        ax.set_xlabel('Data Points')
        ax.set_ylabel('Linear Acceleration {}'.format(['X', 'Y', 'Z'][i]))
        ax.legend()

def window_enumeration(window_arr, data):
    for i, window_size in enumerate(window_arr):
        for j, member_data in enumerate(data):
            visualize_data(member_data[0], member_data[1], member_data[2], member_data[3], window_size)

with h5py.File('main_data.hdf5','r') as hdf:
    data = [get_data('member1'), get_data('member2'), get_data('member3')]
    window_sizes = [5]
    window_enumeration(window_sizes, data)

    features = pd.DataFrame(columns=['mean', 'std', 'max', 'kurtosis', 'skew', 'median', 'range']) #Add shit here
    features['mean'] = data

    scaler = StandardScaler()
    model = LogisticRegression()
    clf = make_pipeline(scaler, model)
    clf.fit(hdf['/model/X_train'], hdf['/model/Y_train'])
    y_pred = clf.predict(hdf['/model/X_train'])
    accuracy = clf.accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
plt.show()

