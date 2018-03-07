#!/usr/bin/env python

'''
Notes:
    - Weak implies weakly supervised learning (4 classes)
    - Strong implies strongly (fully) superversied learning (10 classes)
    - frame number is set to 22ms (default); that is the "sweet spot" based on dsp literature
    - sampling rate is 16kHz (for the MFCC of each track)
'''

import os
import glob
import sys
import time

from processing import mfcc_processing

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

have_mfccs = True

## optimal parameters are passed into the classifers
def svm_classifier(data, test_size, weak=False, verbose=False):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(data['features'])
    labels = data['labels']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)

    if verbose:
        print 'Training sample feature size:', X_train.shape
        print 'Training sample label size:', y_train.shape
        print 'Test sample feature size:' X_test.shape
        print 'Test sample label size:', y_test.shape

    tic = time.time()

    if weak: ## WEAKLY SUPERVISED
        svm_clf = SVC(C=10000, kernel='poly', degree=3, tol=0.0001, max_iter=5000, decision_function_shape='ovr')
        svm_clf.fit(X_train, y_train)
        print svm_clf.score(X_test, y_test)
    else: ## (STRONGLY) SUPERVISED
        svm_clf = SVC(C=10000, kernel='poly', degree=6, tol=0.01, max_iter=5000, decision_function_shape='ovr')
        svm_clf.fit(X_train, y_train)
        print svm_clf.score(X_test, y_test)

    toc = time.time()
    if verbose:
        print 'time it took for SVM classifier to run is', toc-tic
    return

if __name__ == '__main__':
    input_path = './data/genres/'
    mfcc_path = './data/processed/mfcc/'
    mfccs = None
    data = None
    weak = False
    if not have_mfccs:
        have_mfccs = True
        print 'calculating mfccs...'
        mfccs = mfcc_processing.write_mfccs(input_path, mfcc_path, True)
    else :
        print 'retrieving mfccs...'
        mfccs = mfcc_processing.read_mfccs(mfcc_path, True)

    if weak:
        data = mfcc_processing.featurize_data(mfccs, weak=True, verbose=True)
        svm_classifier(data, test_size=0.2, weak=True, verbose=False)
    else:
        data = mfcc_processing.featurize_data(mfccs, weak=False, verbose=True)
        svm_classifier(data, test_size=0.2, weak=False, verbose=False)
