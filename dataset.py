"""
this file contains custom Dataset classes for each of the three datasets used in the paper
"""

from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch as pt
import os
from os.path import join
from skimage.io import imread
from cv2 import resize

to_tensor = ToTensor()


class OCTDataset(Dataset):

    def __init__(self, hdf5, fraction=1):
        """
        opens hdf5 file and obtains list of keys
        :param hdf5: filepath to the hdf5 file
        :param fraction: fraction of keys that will be kept, negative values keep fraction at the end of the list
        """

        # get list of keys from the hdf5 and save filepath as an attribute
        self.hdf5 = hdf5
        self.keys = list(h5py.File(hdf5, 'r').keys())
        self.storage = None

        # keep only specified fraction of keys
        last_index = int(len(self.keys) * abs(fraction))
        self.keys = self.keys[:last_index] if fraction > 0 else self.keys[-last_index:]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):

        # open hdf5 if necessary (due to interaction between h5py and multiprocessing in torch Dataloader)
        if self.storage is None:
            self._open_storage()


        entry = self.storage[self.keys[item]]
        image = to_tensor(np.pad(entry['image'][()], ((0, 0), (12, 12)), mode='reflect'))
        #imlayers = entry['layers'][()]  # segmentations disabled
        label = entry.attrs['label']

        return image, label

    def __del__(self):
        self.close()

    def _open_storage(self):
        self.storage = h5py.File(self.hdf5, 'r')

    def close(self):
        if hasattr(self.storage, 'close'):
            self.storage.close()


class BrainDataset(Dataset):

    def __init__(self, folder, fraction=1, include='all'):
        """
        gather filenames
        :param folder: folder with tif images
        :param fraction: fraction of filenames that will be kept, negative values keep fraction at the end of the list
        :param include: option to filter which images to include: all, pre, flair, post, unique
        """

        # init empty lists
        self.images = []
        self.masks = []

        # parse 'include' filter
        modes = ('pre', 'flair', 'post')
        unique = False
        if include == 'all':
            include = modes
        elif include == 'unique':
            include = modes
            unique = True

        # gather filenames
        for (root, folders, files) in os.walk(folder):
            files = [f for f in files if '.tif' in f and 'mask' not in f]

            # skip if no files in current folder
            if not files:
                continue

            for file in files:
                image = imread(join(root, file)).transpose(2, 0, 1)
                flair = image[1]
                mask = imread(join(root, file.split('.')[0] + '_mask.tif'))
                for i, m in zip(image, modes):
                    if m in include:

                        # skip if 'unique' is set and pre or post is the same as flair
                        if unique and m != 'flair' and (i-flair).sum() == 0:
                            continue

                        self.images.append(i)
                        self.masks.append(mask)

        # keep only specified fraction of samples
        ind = int(fraction * len(self.images))
        if fraction > 0:
            self.images = self.images[:ind]
            self.masks = self.masks[:ind]
        else:
            self.images = self.images[ind:]
            self.masks = self.masks[ind:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = pt.from_numpy(self.images[item].astype(np.float32)/255)[None, ...]
        mask = pt.from_numpy(self.masks[item].astype(np.float32)/255)[None, ...]
        return image, mask


class XRayDataset(Dataset):

    def __init__(self, folder, fraction, size=256):
        """
        gather filenames and load images into memory
        :param folder: folder containing the images
        :param fraction: fraction of filenames that will be kept, negative values keep fraction at the end of the list
        :param size: target size for resizing the images
        """
        self.size = (size, size)
        files = []

        # gather filenames
        for r, ds, fs in os.walk(folder):

            # skip if no files in folder
            if not fs:
                continue

            files += [os.path.join(r, f) for f in fs if '.jpeg' in f]

        self.files = files
        self.images = []
        self.labels = []

        # load and resize images and determine labels
        for f in files:
            self.images.append(resize(imread(f, as_gray=True)/255, self.size))
            if 'bacteria' in f:
                self.labels.append(2)
            elif 'virus' in f:
                self.labels.append(1)
            else:
                self.labels.append(0)

        # keep specified fraction of samples
        ind = int(fraction * len(self.images))
        if fraction > 0:
            self.images = self.images[:ind]
            self.labels = self.labels[:ind]
        else:
            self.images = self.images[ind:]
            self.labels = self.labels[ind:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return pt.from_numpy(self.images[item][None, ...]), self.labels[item]


