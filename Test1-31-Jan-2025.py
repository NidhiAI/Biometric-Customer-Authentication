# -*- coding: utf-8 -*-
"""Updated Speaker Recognition with Metrics"""

import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from FeatureExtraction import extract_features
import warnings
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

warnings.filterwarnings("ignore")

base_path = "../Biometric-Customer-Authentication/"

test_file = os.path.join(base_path, "testDataPath.txt")
file_paths = open(test_file, 'r')
source = os.path.join(base_path, "testData/")
modelpath = os.path.join(base_path, "speakerTrainedModelsGMM/")

gmm_files = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

# Load the GMM Gaussian Models
models = [cPickle.load(open(fname, 'rb')) for fname in gmm_files]
speakers = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]

true_labels = []
pred_labels = []

error = 0
total_sample = 0.0

print("Start with Testing")
take = int(input("Choice 0 or 1: ").strip())
if take == 1:
    path = input("Enter the File name from the sample with .wav notation: ").strip()
    print("Testing Audio File: ", path)
    sr, audio = read(source + path)
    vector = extract_features(audio, sr)
    
    log_likelihood = np.zeros(len(models)) 
    
    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    
    winner = np.argmax(log_likelihood)
    print("The person in the given audio sample is detected as:", speakers[winner])
    time.sleep(1.0)

elif take == 0:
    for path in file_paths:
        total_sample += 1.0
        path = path.strip()
        print("\nTesting Audio File: ", path)
        sr, audio = read(source + path)
        vector = extract_features(audio, sr)
        
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm = models[i]
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
        
        winner = np.argmax(log_likelihood)
        print("\tdetected as - ", speakers[winner])
        
        true_label = path.split("-")[0]  # Assuming filename structure: SpeakerID-Filename.wav
        true_labels.append(true_label)
        pred_labels.append(speakers[winner])
        
        if speakers[winner] != true_label:
            error += 1
        
        time.sleep(1.0)
    
    print('\n\nTotal number of audios tested: ', total_sample)
    print('Total users not identified correctly: ', error)
    
    accuracy = accuracy_score(true_labels, pred_labels)
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels)
    
    print("MFCC + GMM current testing Accuracy in Percentage: ", accuracy * 100, "%")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(report)

print("Speaker Identification process End")
