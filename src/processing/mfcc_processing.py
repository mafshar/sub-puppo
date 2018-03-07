import os
import glob
import time
import numpy as np

from librosa import load
from librosa import feature
from collections import defaultdict

WEAK_CLASSES = ['classical', 'jazz', 'metal', 'pop']

def calc_mfcc(file):
    y, sr = load(file, sr=16050)
    return feature.mfcc(y=y, sr=sr, n_mfcc=22)

def write_mfccs(input_path, output_path, verbose=False):
    mfccs = defaultdict(lambda: [])
    for sub in os.listdir(input_path):
        input_dir = os.path.join(input_path, sub)
        output_dir = os.path.join(output_path, sub)
        tic = time.time()
        if verbose:
            print 'generating mfccs for', sub
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for input_file in glob.glob(os.path.join(input_dir, '*.au')):
            count = os.path.basename(input_file).split('.')[1]
            output_file = os.path.join(output_dir, sub + '_' + count + '_mfcc.npy')
            mfcc = calc_mfcc(input_file)
            print mfcc.shape
            mfccs[sub].append(mfcc)
            np.save(file=output_file, arr=mfcc)
        toc = time.time()
        if verbose:
            print '\ttime it took to obtain mfccs for', sub, 'is', toc-tic
    return mfccs

def read_mfccs(path, verbose=False):
    mfccs = defaultdict(lambda: [])
    for sub in os.listdir(path):
        dir = os.path.join(path, sub)
        if not os.path.exists(dir):
            print 'directory not found, exiting...'
            exit(-1)
        tic = time.time()
        if verbose:
            print 'retrieving mfccs for', sub
        for file in glob.glob(os.path.join(dir, '*.npy')):
            mfccs[sub].append(np.load(file))
        toc = time.time()
        if verbose:
            print '\ttime it took to retrieve mfccs for', sub, 'is', toc-tic
    return mfccs

def featurize_data(mfccs, weak=False, verbose=False):
    data = {}
    features = []
    labels = []
    label = 0
    for genre in mfccs:
        if weak and genre not in WEAK_CLASSES:
            continue
        label += 1
        tic = time.time()
        if verbose:
            print 'calculating mfcc features for', genre
        for mfcc in mfccs[genre]:
            features.append(np.sum(mfcc, axis=1, keepdims=False))
            labels.append(label)
        toc = time.time()
        if verbose:
            print '\ttime taken to calculate mfcc features for', genre, 'is', toc-tic
    data['features'] = np.array(features)
    data['labels'] = np.array(labels)
    return data
