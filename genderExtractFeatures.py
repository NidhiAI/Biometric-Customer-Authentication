# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kaf6j0NuJcKJ9i3CHKlTuggzKCjji_Kg
"""

import os
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import os
from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras

from keras import layers

#from sklearn.preprocessing import LabelEncoder#, StandardScalerimport keras
#from keras.models import Sequentialimport #, warnings
#warnings.filterwarnings('ignore')


# create headers 
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

base_path = "../Biometric-Customer-Authentication/"

#import csv
#csvFilePath = "/content/Speaker-Recognition-GMM/maleFemale/"
csvFilePath = (os.path.join(base_path,"genderDetection/")) 
file = open(csvFilePath+'maleFemaleAudioFeatures.csv', 'w', newline='')
audioFilePath = (os.path.join(csvFilePath,"genderDetectionTrainingData/")) 

with file:
    writer = csv.writer(file)
    writer.writerow(header)
#genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
genres = 'male female'.split()
for g in genres:
    #for filename in os.listdir(f'/content/Speaker-Recognition-GMM/maleFemale/maleFemaleAudio/{g}'):
    #    songname = f'/content/Speaker-Recognition-GMM/maleFemale/maleFemaleAudio/{g}/{filename}'
    for filename in os.listdir(f'{audioFilePath}{g}'):
        songname = f'{audioFilePath}{g}/{filename}'    
        y, sr = librosa.load(songname, mono=True, duration=30)
        #rmse = librosa.feature.rmse(y=y)
        rmse = librosa.feature.rms(y=y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open(csvFilePath+'maleFemaleAudioFeatures.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())