# Author(s): Ryan Silverberg
import numpy as np
import h5py

# Data Processing

with h5py.File('/main_data.h5','w') as hdf:
    g1 = hdf.create_group('/group1')