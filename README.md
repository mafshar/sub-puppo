# Music Genre Classification

Mohammad Afshar, ma2510@nyu.edu

## Overview

### Background and Motivation

TBA

## Dataset

Marsyas, which stands for "Music Analysis, Retrieval, and Synthesis for Audio Signals," is an open source software framework for audio processing with an emphasis on Music Information Retrieval Applications (MIRA). Additionally, the site also gives access to certain datasets. One particular dataset is the GTZAN Genre Collection, which contains 1000 audio tracks, each 30 seconds long, covering 10 genres; therefore, there are 100 tracks per genre. All tracks are 22050Hz, Mono 16-bit audio files in .au format. This project covers both weakly supervised learning models (covering classes of genres), as well as fully supervised learning models for the entire dataset. The reason for the distinction is that as the number of classes increases, the size of the number of samples also needs to increase, otherwise there will be a sharp drop in the accuracy (see previous works in audio and signal processing, particularly in MIRA).

### Classes (Genres)

The full dataset is as follows (the classes with a \* next to them are the ones that are used for the weakly supervised learning models)

1. blues
2. classical *
3. country
4. disco
5. hiphop
6. jazz *
7. metal *
8. pop *
9. reggae
10. rock

## Methodology
TBA

## Model

Three models explored:
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Neural Network (NN)

## Results

### SVM

Weakly supervised: 0.875
Fully supervised:  0.61

### KNN

Weakly supervised: 0.9
Fully supervised:  0.63

## Setup

### Procuring the Data

Create a `data` directory in the project root directory and type in the following commands:
```bash
$ wget http://opihi.cs.uvic.ca/sound/genres.tar.gz -P ./data
```

To run the code:
```bash
$ python ./src/main.py
```
