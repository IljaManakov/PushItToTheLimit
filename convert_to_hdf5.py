"""
script for converting the OCT dataset from matlab format to a hdf5 file containing numpy arrays
"""

import h5py
from scipy.io import loadmat
import os
import numpy as np

# gather filenames and initialize hdf5 storage
folder = ''  # base folder with data samples
storage = h5py.File(os.path.join(folder, 'dataset.hdf5'), 'w', swmr=True, libver='latest')

files = []
files += [(os.path.join(folder, 'control',  f), 0) for f in os.listdir(os.path.join(folder, 'control'))]
files += [(os.path.join(folder, 'amd', f), 1) for f in os.listdir(os.path.join(folder, 'amd'))]

# iteratively convert matlab files to numpy as store in hdf5
for i, (file, label) in enumerate(files):

    # convert
    images = loadmat(file)
    layers = images['layerMaps'].transpose((0, 2, 1))
    images = images['images'].transpose(2, 0, 1)

    # write individual b-scans to storage (for easier batching during training)
    for j, (image, layer) in enumerate(zip(images, layers)):

        group = storage.require_group(f'{i}-{j}')
        group['image'] = image
        if len(layer[~np.isnan(layer)]) > 0:
            group['layers'] = layer.astype(np.int16)
        else:
            group['layers'] = 0
        group.attrs['label'] = label
        storage.flush()
        print(f'{100*i+j} out of {100*len(files)}')

# flush storage
storage.close()

