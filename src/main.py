#!/usr/bin/env python

'''
Notes:
    - Weak implies weakly supervised learning (4 classes)
    - Strong implies strongly (fully) superversied learning (10 classes)
    - frame number is set to 20ms (default); that is the "sweet spot" based on dsp literature
    - sampling rate is 16kHz (for the MFCC of each track)
'''

import os
import glob
import sys
from processing import mfcc_processing

genres = ['pop', 'rock', 'hiphop', 'jazz', 'metal', 'disco', 'blues', 'reggae', 'country', 'classical']
have_mfccs = True

if __name__ == '__main__':
    input_path = './data/genres/'
    mfcc_path = './data/processed/mfcc/'
    mfccs = None
    features = None
    if not have_mfccs:
        have_mfccs = True
        print 'calculating mfccs...'
        mfccs = mfcc_processing.write_mfccs(input_path, mfcc_path, True)
    else :
        print 'retrieving mfccs...'
        mfccs = mfcc_processing.read_mfccs(mfcc_path, True)

    features = mfcc_processing.calc_features(mfccs, True)

    for genre in features:
        for mfcc in features[genre]:
            print mfcc.shape
