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

import numpy as np

from processing import mfcc_processing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize

input_path = './data/genres/'
mfcc_path = './data/processed/mfcc/'
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

def svm_classifier(data, test_size, weak=False, verbose=False):
    norm_data = normalize_and_split(data, test_size, verbose)
    X_train = norm_data['X_train']
    X_test = norm_data['X_test']
    y_train = norm_data['y_train']
    y_test = norm_data['y_test']

    tic = time.time()

    svm_clf = SVC(C=10000, kernel='poly', degree=3, tol=0.0001, max_iter=5000, decision_function_shape='ovr') if weak \
        else SVC(C=10000, kernel='poly', degree=6, tol=0.01, max_iter=5000, decision_function_shape='ovr')
    svm_clf.fit(X_train, y_train)
    print 'TEST ACCURACY:', svm_clf.score(X_test, y_test)

    toc = time.time()
    if verbose:
        print '\ttime taken  for SVM classifier to run is', toc-tic
    return

def knn_classifier(data, test_size, weak=False, verbose=False):
    norm_data = normalize_and_split(data, test_size, verbose)
    X_train = norm_data['X_train']
    X_test = norm_data['X_test']
    y_train = norm_data['y_train']
    y_test = norm_data['y_test']

    tic = time.time()

    knn_clf = KNeighborsClassifier(n_neighbors=3, weights='distance', p=1, n_jobs=-1) if weak \
        else KNeighborsClassifier(n_neighbors=8, weights='distance', p=1, n_jobs=-1)
    knn_clf.fit(X_train, y_train)
    print 'TEST ACCURACY:', knn_clf.score(X_test, y_test)

    toc = time.time()
    if verbose:
        print '\ttime taken  for KNN classifier to run is', toc-tic
    return

def build_vocabulary(data, verbose=False):
    parameters = {}
    features = data['features']
    labels = data['labels']
    kmeans = KMeans(n_clusters=5, init='random', max_iter=500, n_jobs=-1)
    kmeans.fit(features)
    vocab = kmeans.cluster_centers_
    dist_features = kmeans.transform(features)
    parameters['dist_features'] = dist_features
    parameters['labels'] = labels
    parameters['vocab'] = vocab
    return parameters

## @TODO: write the function that creates a feature vector for each of the songs
def generate_histogram_features(data, parameters, verbose=False):
    vocab = parameters['vocab']
    dist_features = parameters['dist_features']
    min_indices = np.argmin(dist_features, axis=1)
    features = []
    print min_indices
    return

## @TODO: build classifier using the generate_histogram_features function
## bag of audio words approach (histogram of occurences)
def bag_of_words_classifier(data, test_size, weak=False, verbose=False):
    parameters = build_vocabulary(data, verbose)

    histo_data = generate_histogram_features(data, parameters, verbose)

    norm_data = normalize_and_split(histo_data, test_size, verbose)
    X_train = norm_data['X_train']
    X_test = norm_data['X_test']
    y_train = norm_data['y_train']
    y_test = norm_data['y_test']

    tic = time.time()
    toc = time.time()
    if verbose:
        print 'time taken  for BoW Classification to run is', toc-tic
    return

if __name__ == '__main__':

    mfccs = None
    data = None

    if not have_mfccs:
        have_mfccs = True
        print 'calculating mfccs...'
        mfccs = mfcc_processing.write_mfccs(input_path, mfcc_path, True)
    else :
        print 'retrieving mfccs...'
        mfccs = mfcc_processing.read_mfccs(mfcc_path, True)


    data = mfcc_processing.featurize_data(mfccs, weak=True, verbose=True)
    # params = build_vocabulary(data, verbose=True)
    # generate_histogram_features(data, params, verbose=True)

    print

    weak = False
    if weak:
        data = mfcc_processing.featurize_data(mfccs, weak=True, verbose=True)
        print
        svm_classifier(data, test_size=0.10, weak=True, verbose=True)
        print
        knn_classifier(data, test_size=0.10, weak=True, verbose=True)
    else:
        data = mfcc_processing.featurize_data(mfccs, weak=False, verbose=True)
        print
        svm_classifier(data, test_size=0.10, weak=False, verbose=True)
        print
        knn_classifier(data, test_size=0.10, weak=False, verbose=True)
