# -*- coding: utf-8 -*-
"""

"""

#libraries
import os
import pickle
import numpy as np
from pyAudioAnalysis import audioBasicIO as aIO 
from pyAudioAnalysis import MidTermFeatures as aF
    
# load data scaler and model 
with open("scaler.pickle", "rb") as f: 
    scaler = pickle.load(f)
with open("model.pickle", "rb") as f: 
    model, selected_indices = pickle.load(f)

# extract features for the wav file at 0.5, 0.25, 0.05, 0.05
mid_window, mid_step, short_window, short_step = 0.5, 0.25, 0.05, 0.05

wav_file_dir = os.listdir("test_sample")
fs, s = aIO.read_audio_file("test_sample/{}".format(wav_file_dir[0]))

mt, st, feature_names = aF.mid_feature_extraction(s, fs, 
                                                mid_window * fs, mid_step * fs, 
                                                short_window * fs, short_step * fs)
file_features = 0
for j in range(mt.shape[1]):
    file_features += mt[:,j]
file_features = file_features / mt.shape[1] # shape is (136,)
#print(file_features)

# X_test_unscaled
X_test_unscaled = np.reshape(file_features, (1,file_features.shape[0]))
# scale data with scaler learned from Crema training dataset
X_test_scaled = scaler.transform(X_test_unscaled)
# feature selection - use selected_indices only
X_test = X_test_scaled[:, selected_indices]

#make prediction
y_pred = model.predict(X_test)  #shape is (no_of_labels, )

if y_pred == 0:
    print('The predicted emotion is "happy"')
elif y_pred == 1:
    print('The predicted emotion is "neutral"')
elif y_pred == 2:
    print('The predicted emotion is "sad"')
else : #if y_pred == 3
    print('The predicted emotion is "angry"')