#!/usr/bin/env python

'''
Notes:
    - Weak implies weakly supervised learning (4 classes)
    - Strong implies strongly (fully) superversied learning (10 classes)
    - frame number is set to 20ms (default); that is the "sweet spot" based on dsp literature
'''

import os
import glob
import sys
import time
import numpy as np

from librosa import load
from librosa import feature
from collections import defaultdict

GENRES = ['pop', 'rock', 'hiphop', 'jazz', 'metal', 'disco', 'blues', 'reggae', 'country', 'classical']
CALCULATED_MFCCS = True

def calc_mfccs(input_path, output_path, verbose=False):
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
            y, sr = load(input_file)
            mfcc = feature.mfcc(y=y, sr=sr)
            mfccs[sub].append(mfcc)
            np.save(file=output_file, arr=mfcc)
        toc = time.time()
        if verbose:
            print '\ttime it took to obtain mfccs for', sub, 'is', toc-tic
    return mfccs

def retrieve_mfccs(path, verbose=True):
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

if __name__ == '__main__':
    input_path = './data/genres/'
    mfcc_path = './data/processed/mfcc'
    mfccs = None
    if not CALCULATED_MFCCS:
        print 'calculating mfccs...'
        mfccs = calc_mfccs(input_path, mfcc_path, True)
    else :
        print 'retrieving mfccs...'
        mfccs = retrieve_mfccs(mfcc_path)

    ## they are of different sizes!!!
    #@TODO: come up with a way to allow for learning given a set of mfccs
    for genre in mfccs:
        for coeffs in mfccs[genre]:
            print coeffs.shape
