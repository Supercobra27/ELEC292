# Example of how to work with hdf5 files

import numpy as np
import h5py

m1 = np.random.random(size=(1000,1000))
m2 = np.random.random(size=(1000,1000))

with h5py.File('./hdf5_data.h5','w') as hdf:
    g1 = hdf.create_group('/group1')
    g1.create_dataset('dataset1', data=m1)
    g1.create_dataset('dataset2', data=m2)
    g2 = hdf.create_group('/group2/hehehaha')
    g2.create_dataset('dataset3', data=m1)

with h5py.File('./hdf5_data.h5','r') as hdf:
    ls = list(hdf.keys())
    print(ls)
    d1 = hdf.get('dataset1')
    print(type(d1))
    a1 = np.array(d1)
    print(type(a1))
    d2 = g2.get('dataset3')
    d2 = np.array(d2)
    print(d2.shape)
    d3 = hdf.get('/group1')
    print(list(d3.items()))
