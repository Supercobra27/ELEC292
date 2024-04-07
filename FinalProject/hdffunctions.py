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

WALKING = 0
JUMPING = 1

# Data Processing
if os.path.exists("main_data.hdf5"):
    os.remove("main_data.hdf5")

def segment(dataset, name, member):
    i = 0
    j = 0
    with h5py.File('main_data.hdf5', 'a') as hdf:
        tr = hdf.create_group('/model/train/'+member)
        while i < len(dataset):
            while j <= len(dataset) and dataset[1][j] - dataset[1][i] < 5:
                j += 1
            tr.create_dataset(name+str(j), data=dataset.iloc[i:j, :])
            i += 1

def process_data(member_name, walk_csv, jump_csv):
    walk_data = LabelData(walk_csv, 'walking')
    jump_data = LabelData(jump_csv, 'jumping')
    with h5py.File('main_data.hdf5', 'a') as hdf:
        mg = hdf.create_group('/'+member_name)
        mg.create_dataset('walk', data=walk_data)
        mg.create_dataset('jump', data=jump_data)
        return walk_data, jump_data
    
def LabelData(fileName, label):
    data = pd.read_csv(fileName)
    data = pd.DataFrame(data)
    if label == 'walking':
        data.insert(0, "label", WALKING)
    if label == 'jumping':
        data.insert(0, "label", JUMPING)
    return data

def get_data(group_name):
    with h5py.File('main_data.hdf5', 'r') as hdf:
        walk_data = pd.DataFrame(np.array(hdf['/'+group_name+'/walk']))
        jump_data = pd.DataFrame(np.array(hdf['/'+group_name+'/jump']))
        return walk_data, jump_data
    
def get_segment(segment_name, member_name):
    with h5py.File('main_data.hdf5', 'r') as hdf:
        return pd.DataFrame(np.array(hdf['/model/train/'+member_name+'/'+segment_name]))