#!/usr/bin/env python

'''
Notes:
    - Weak implies weakly supervised learning (4 classes)
    - Strong implies strongly (fully) superversied learning (10 classes)
    - frame number is set to 22ms (default); that is the "sweet spot" based on dsp literature
    - sampling rate is 16kHz (for the MFCC of each track)
    - Accuracy increases as the test set gets smaller, which  implies that a lot of these machine learning models are heavily data-driven (i.e. feed more data for more performance boosts)
    - Currently, optimal benchmark results are achieved with a test set size of 10 percent of the total data
'''

import os
import glob
import sys
import time

from processing import mfcc_processing

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize

have_mfccs = True

def normalize_and_split(data, test_size, verbose=False):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(data['features'])
    labels = data['labels']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    norm_data = {}
    norm_data['X_train'] = X_train
    norm_data['X_test'] = X_test
    norm_data['y_train'] = y_train
    norm_data['y_test'] = y_test
    if verbose:
        print 'Training sample feature size:', X_train.shape
        print 'Training sample label size:', y_train.shape
        print 'Test sample feature size:', X_test.shape
        print 'Test sample label size:', y_test.shape
    return norm_data

## optimal parameters are passed into the classifers
def svm_classifier(data, test_size, weak=False, verbose=False):
    norm_data = normalize_and_split(data, test_size, verbose)
    X_train = norm_data['X_train']
    X_test = norm_data['X_test']
    y_train = norm_data['y_train']
    y_test = norm_data['y_test']

    tic = time.time()

    if weak: ## WEAKLY SUPERVISED (Top Accuracy at 85%)
        svm_clf = SVC(C=10000, kernel='poly', degree=3, tol=0.0001, max_iter=5000, decision_function_shape='ovr')
        svm_clf.fit(X_train, y_train)
        print svm_clf.score(X_test, y_test)
    else: ## (STRONGLY) SUPERVISED (Top Accuracy at 60%)
        svm_clf = SVC(C=10000, kernel='poly', degree=6, tol=0.01, max_iter=5000, decision_function_shape='ovr')
        svm_clf.fit(X_train, y_train)
        print svm_clf.score(X_test, y_test)

    toc = time.time()
    if verbose:
        print 'time it took for SVM classifier to run is', toc-tic
    return

def knn_classifier(data, test_size, weak=False, verbose=False):
    norm_data = normalize_and_split(data, test_size, verbose)
    X_train = norm_data['X_train']
    X_test = norm_data['X_test']
    y_train = norm_data['y_train']
    y_test = norm_data['y_test']

    tic = time.time()

    if weak: ## WEAKLY SUPERVISED (Top Accuracy at 83.75%)
        knn_clf = KNeighborsClassifier(n_neighbors=3, weights='distance', p=1, n_jobs=-1)
        knn_clf.fit(X_train, y_train)
        print knn_clf.score(X_test, y_test)
    else: ## (STRONGLY) SUPERVISED (Top Accuracy at 57.5%)
        knn_clf = KNeighborsClassifier(n_neighbors=8, weights='distance', p=1, n_jobs=-1)
        knn_clf.fit(X_train, y_train)
        print knn_clf.score(X_test, y_test)

    toc = time.time()
    if verbose:
        print 'time it took for KNN classifier to run is', toc-tic

    return

if __name__ == '__main__':
    input_path = './data/genres/'
    mfcc_path = './data/processed/mfcc/'
    mfccs = None
    data = None

    if not have_mfccs:
        have_mfccs = True
        print 'calculating mfccs...'
        mfccs = mfcc_processing.write_mfccs(input_path, mfcc_path, True)
    else :
        print 'retrieving mfccs...'
        mfccs = mfcc_processing.read_mfccs(mfcc_path, True)

    weak = False
    if weak:
        data = mfcc_processing.featurize_data(mfccs, weak=True, verbose=True)
        svm_classifier(data, test_size=0.10, weak=True, verbose=True)
        knn_classifier(data, test_size=0.10, weak=True, verbose=True)
    else:
        data = mfcc_processing.featurize_data(mfccs, weak=False, verbose=True)
        svm_classifier(data, test_size=0.10, weak=False, verbose=True)
        knn_classifier(data, test_size=0.10, weak=False, verbose=True)
