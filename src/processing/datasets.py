
import os
import glob

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from mfcc_processing import featurize_mfcc

import warnings
warnings.filterwarnings('ignore')

WEAK = {'classical': 0, 'jazz': 1, 'metal':2, 'pop': 3}
FULLY = {'classical': 0, 'jazz': 1, 'metal':2, 'pop': 3, 'blues': 4, 'country': 5, 'disco': 6, 'hiphop': 7, 'reggae': 8, 'rock': 9}

def train_test_dataset_split(dataset):
    num_train = len(dataset)
    indices = list(range(num_train))
    split = 4

    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = DataLoader(dataset, batch_size=10, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=4, sampler=validation_sampler)
    return train_loader, validation_loader


class ToTensor(object):
    '''Convert ndarrays in sample to Tensors'''
    def __call__(self, sample):
        return torch.from_numpy(sample).float()


class MfccDatasetWeak(Dataset):

    def __init__(self, mfcc_path, transform=None):
        self.mfcc_path = mfcc_path
        self.transform = transform
        self.datafiles =  self.get_file_paths(self.mfcc_path)

    def get_file_paths(self, path):
        datafiles = []
        for sub in os.listdir(path):
            if sub not in WEAK.keys():
                continue
            dir = os.path.join(path, sub)
            if not os.path.exists(dir):
                print 'directory not found, exiting...'
                exit(-1)
            for file in glob.glob(os.path.join(dir, '*.npy')):
                datafiles.append((file, sub))
        return datafiles

    def __len__(self):
        return len(self.datafiles)

    def __getitem__(self, idx):
        mfcc = featurize_mfcc(
            np.load(self.datafiles[idx][0]))
        genre = WEAK[self.datafiles[idx][1]]

        if self.transform:
            mfcc = self.transform(mfcc)

        return (mfcc, genre)

class MfccDataset(Dataset):

    def __init__(self, mfcc_path, transform=None):
        self.mfcc_path = mfcc_path
        self.transform = transform
        self.datafiles =  self.get_file_paths(self.mfcc_path)

    def get_file_paths(self, path):
        datafiles = []
        for sub in os.listdir(path):
            dir = os.path.join(path, sub)
            if not os.path.exists(dir):
                print 'directory not found, exiting...'
                exit(-1)
            for file in glob.glob(os.path.join(dir, '*.npy')):
                datafiles.append((file, sub))
        return datafiles

    def __len__(self):
        return len(self.datafiles)

    def __getitem__(self, idx):
        mfcc = featurize_mfcc(
            np.load(self.datafiles[idx][0]))
        genre = FULLY[self.datafiles[idx][1]]

        if self.transform:
            mfcc = self.transform(mfcc)

        return (mfcc, genre)
