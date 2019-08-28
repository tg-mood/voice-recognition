import os
import pandas as pd
import librosa
import glob
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
from keras.utils import np_utils
from sklearn import preprocessing

lb = preprocessing.LabelEncoder()

feeling_list=[]
feeling_list.append('female_calm')
feeling_list.append('female_happy')
feeling_list.append('female_sad')
feeling_list.append('female_angry')
feeling_list.append('female_fearful')
feeling_list.append('male_calm')
feeling_list.append('male_happy')
feeling_list.append('male_sad')
feeling_list.append('male_angry')
feeling_list.append('male_fearful')

labels = pd.DataFrame(feeling_list)

test = np.array(labels)
test = np_utils.to_categorical(lb.fit_transform(test))

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("voice_detection_model.h5")

data, sampling_rate = librosa.load('audio_file.wav')
plt.figure(figsize=(15, 5))

X, sample_rate = librosa.load('audio_file.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
featurelive = mfccs
livedf2 = featurelive

livedf2= pd.DataFrame(data=livedf2)
livedf2 = livedf2.stack().to_frame().T
livedf2
twodim= np.expand_dims(livedf2, axis=2)

livepreds = loaded_model.predict(twodim,
                         batch_size=32,
                         verbose=1)

livepreds1=livepreds.argmax(axis=1)

liveabc = livepreds1.astype(int).flatten()

livepredictions = (lb.inverse_transform((liveabc)))

print(livepredictions)