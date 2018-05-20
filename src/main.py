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
import warnings
warnings.filterwarnings("ignore")

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from processing import mfcc_processing, datasets
from deep_models import models
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

def mfcc_nn_model(num_epochs, test_size, weak=False, verbose=False):
    tic = time.time()

    tensorize = datasets.ToTensor()
    dataset = None
    net = None

    if weak:
        dataset = datasets.MfccDatasetWeak(mfcc_path, tensorize)
        net = models.MfccNetWeak()
    else:
        dataset = datasets.MfccDataset(mfcc_path, tensorize)
        net = models.MfccNet()

    trainloader, testloader = datasets.train_test_dataset_split(dataset)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.8)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if verbose and i % 5 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))

            running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
                inputs, labels = data
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    print 'TEST ACCURACY:', 1. * correct / total

    toc = time.time()
    if verbose:
        print '\ttime taken  for Mfcc NN to run is', toc-tic

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

    print

    weak = False
    if weak:
        data = mfcc_processing.featurize_data(mfccs, weak=True, verbose=True)
        print
        svm_classifier(data, test_size=0.10, weak=True, verbose=True)
        print
        knn_classifier(data, test_size=0.10, weak=True, verbose=True)
        print
        mfcc_nn_model(num_epochs=10, test_size=0.10, weak=True, verbose=True)
    else:
        data = mfcc_processing.featurize_data(mfccs, weak=False, verbose=True)
        print
        svm_classifier(data, test_size=0.10, weak=False, verbose=True)
        print
        knn_classifier(data, test_size=0.10, weak=False, verbose=True)
        print
        mfcc_nn_model(num_epochs=10, test_size=0.10, weak=False, verbose=True)
