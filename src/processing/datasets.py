import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import warnings
warnings.filterwarnings('ignore')

WEAK = {'classical': 0, 'jazz': 1, 'metal':2, 'pop': 4}

class MfccDatasetWeak(Dataset):

    def __init__(self, mfcc_path, transform=None):
        self.mfcc_path = mfcc_path
        self.transform = transform
        self.datafiles = self.get_file_paths(self.mfcc_path)

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
        return len(datafiles)

    def __getitem__(self, idx):
        mfcc = np.load(datafiles[idx][0])
        genre = WEAK[datafiles[idx][1]]
        sample = {'mfcc': mfcc, 'genre': genre}

        if self.transform:
            sample = self.transform(sample)

        return sample


def train_test_split(dataset):
    num_train = len(dataset)
    indices = list(range(num_train))
    split = 4

    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=2, sampler=validation_sampler)
