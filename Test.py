# -*- coding: utf-8 -*-
"""Untitled32.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1o3oHA7496-4BBjvriUlErVZcSU1WUH7j
"""

import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from FeatureExtraction import extract_features
import warnings
warnings.filterwarnings("ignore")
import time

base_path = "../Biometric-Customer-Authentication/"

#path to get test data files name
#test_file = "/content/Speaker-Recognition-GMM/testDataPath.txt"  
test_file   = (os.path.join(base_path,"testDataPath.txt"))     
file_paths = open(test_file,'r')

#path to read test data wav files
#source   = "/content/Speaker-Recognition-GMM/testData/"   
source   = (os.path.join(base_path,"testData/")) 

#path where training speakers GMM modles are saved
#modelpath =  "/content/Speaker-Recognition-GMM/speakerTrainedModelsGMM/"
modelpath = (os.path.join(base_path,"speakerTrainedModelsGMM/"))
##"Trained_Speech_Models/"

gmm_files = [os.path.join(modelpath,fname) for fname in 
              os.listdir(modelpath) if fname.endswith('.gmm')]

#Load the GMM Gaussian Models
models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
              in gmm_files]

error = 0
total_sample = 0.0

print(" Start with Testing ")
#print("Press '1' for testing a single Audio or Press '0' for testing complete set of audio with Accuracy?")
print("\nPress '1' for testing single Audio or Press '0' for testing full list of audios ?")
#take=int(input().strip())
take=int(input("Choice 0 or 1: ").strip())
if take == 1:
    print ("Enter the File name from the sample with .wav notation :")
    path =input().strip()
    print (("Testing Audio File : ",path))
    sr,audio = read(source + path)
    vector   = extract_features(audio,sr)
    
    log_likelihood = np.zeros(len(models)) 
    
    for i in range(len(models)):
        gmm    = models[i]  #checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    
    winner = np.argmax(log_likelihood)
    #print ("\tThe person in the given audio sample is detected as - ", speakers[winner])
    print ("\nThe person in the given audio sample is detected as : ", speakers[winner])

    time.sleep(1.0)
elif take == 0:
   # test_file = "/content/Speaker-Recognition-GMM/testDataPath.txt" 
    test_file = (os.path.join(base_path,"testDataPath.txt"))        
    file_paths = open(test_file,'r')
    # Read the test directory and get the list of test audio files 
    for path in file_paths:   
        total_sample+= 1.0
        path=path.strip()
        print("\nTesting Audio File : ", path)
        sr,audio = read(source + path)
        vector   = extract_features(audio,sr)
        log_likelihood = np.zeros(len(models)) 
        for i in range(len(models)):
            gmm    = models[i]  #checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
        winner=np.argmax(log_likelihood)
        print ("\tdetected as - ", speakers[winner])
        #checker_name = path.split("_")[0]
        checker_name = path.split("-")[0]
        if speakers[winner] != checker_name:
            error += 1
        time.sleep(1.0)
   # print (error, total_sample)
    print ('\n\nTotal number of audios tested:  ',total_sample)
    print ('Total users not identified correctly:  ',error)
    
    accuracy = ((total_sample - error) / total_sample) * 100

    print ("MFCC + GMM current testing Accuracy in Percentage :  ", accuracy, "%")


#print ("Speaker Identified Successfully")
print ("Speaker Identification process End")
