# Author(s): Ryan Silverberg
import numpy as np
import h5py
import pandas as pd
from matplotlib import pyplot as plt

# Data Processing

# Read Walk CSVs
m1walk = pd.read_csv("RyanWalk.csv")
m2walk = pd.read_csv("FosterWalk.csv")
m3walk = pd.read_csv("JuliaWalk.csv")

# Read Jump CSVs
m1jump = pd.read_csv("RyanJump.csv")
m2jump = pd.read_csv("FosterJump.csv")
m3jump = pd.read_csv("JuliaJump.csv")

with h5py.File('main_data.hdf5','w') as hdf:

    # Create Empty Model Groups
    g1 = hdf.create_group('/model')
    g1.create_dataset('train', dtype="f8")
    g1.create_dataset('test', dtype="f8")

    # Create Member Groups
    m1g = hdf.create_group('/member1')
    m2g = hdf.create_group('/member2')
    m3g = hdf.create_group('/member3')

    # Create Walk Datasets
    m1g.create_dataset('walk', data=m1walk)
    m2g.create_dataset('walk', data=m2walk)
    m3g.create_dataset('walk', data=m3walk)

    # Create Jump Datasets
    m1g.create_dataset('jump', data=m1jump)
    m2g.create_dataset('jump', data=m2jump)
    m3g.create_dataset('jump', data=m3jump)


    
with h5py.File('main_data.hdf5','r') as hdf:
    # Read Walk CSVs
    m1w = pd.DataFrame(np.array(hdf['/member1/walk']))
    m2w = pd.DataFrame(np.array(hdf['/member2/walk']))
    m3w = pd.DataFrame(np.array(hdf['/member3/walk']))

    # Read Jump CSVs
    m1j = pd.DataFrame(np.array(hdf['/member1/jump']))
    m2j = pd.DataFrame(np.array(hdf['/member2/jump']))
    m3j = pd.DataFrame(np.array(hdf['/member3/jump']))
    # Need to add JUMP

    # Time Column
    x_in_m1 = m1w.iloc[:, 0]
    x_in_m2 = m2w.iloc[:, 0]
    x_in_m3 = m3w.iloc[:, 0]

    # Array of X, Y, Z Acceleration
    data_m1 = [m1w.iloc[:, 1], m1w.iloc[:, 2], m1w.iloc[:, 3]]
    data_m2 = [m2w.iloc[:, 1], m2w.iloc[:, 2], m2w.iloc[:, 3]]
    data_m3 = [m3w.iloc[:, 1], m3w.iloc[:, 2], m3w.iloc[:, 3]]


    fig, ax = plt.subplots(figsize=(10,10))

    window = 31

    m1_graph = [data_m1[0].rolling(window).mean(), data_m1[1].rolling(window).mean(), data_m1[2].rolling(window).mean()]
    m2_graph = [data_m2[0].rolling(window).mean(), data_m2[1].rolling(window).mean(), data_m2[2].rolling(window).mean()]
    m3_graph = [data_m3[0].rolling(window).mean(), data_m3[1].rolling(window).mean(), data_m3[2].rolling(window).mean()]

    ax.plot(x_in_m1, m1_graph[0], linewidth=1)
    ax.plot(x_in_m2, m2_graph[0], linewidth=1)
    ax.plot(x_in_m3, m3_graph[0], linewidth=1)
    ax.set_xlabel('Data Points')
    ax.set_ylabel('Linear Acceleration X')

    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(x_in_m1, m1_graph[1], linewidth=1)
    ax.plot(x_in_m2, m2_graph[1], linewidth=1)
    ax.plot(x_in_m3, m3_graph[1], linewidth=1)
    ax.set_xlabel('Data Points')
    ax.set_ylabel('Linear Acceleration Y')
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(x_in_m1, m1_graph[2], linewidth=1)
    ax.plot(x_in_m2, m2_graph[2], linewidth=1)
    ax.plot(x_in_m3, m3_graph[2], linewidth=1)
    ax.set_xlabel('Data Points')
    ax.set_ylabel('Linear Acceleration Z')
    plt.show()

