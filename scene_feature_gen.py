import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import librosa.display
import numpy as np 
import pandas as pd
import os
import re
from sklearn.utils import shuffle
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import random
import librosa
import pickle
WINDOW_SLICE = 10 
TEST_AUDIO_PATH =  "test_audios/"
TEST_FEATURE_BATCH_PATH =  "test_feature_batch/"
Scene_Change = {}
NUM_OF_BANDS = 200

   
def generate_spec( recording, OFFSET ):
  
	audio, _ = librosa.core.load(recording, sr=44100, dtype=np.float32, duration=10.0, offset  = OFFSET )

	mean_audio = audio.mean() 
	std_audio = np.std(audio) 
	audio  = ( audio - mean_audio )/std_audio 

	spec = librosa.feature.melspectrogram(audio, sr=44100, n_fft=2205, hop_length=882,
											  n_mels=NUM_OF_BANDS, fmax=22050, power=2)
	spec = librosa.power_to_db(spec)

	mean_spec = spec.mean()  
	std_spec = np.std(spec) 
	spec = ( spec - mean_spec )/std_spec 
	# print( " MEl spectro gram for the audio starting from "  + str( OFFSET )  + " ending 10 seconds later"  )   
	# plt.figure(figsize=(10, 4))
	# librosa.display.specshow( spec , x_axis= 'time', y_axis= 'mel',fmax = 8000 )
	# plt.colorbar(format='%+2.0f dB')
	# plt.title('Mel spectrogram')
	# plt.tight_layout()
	# plt.show()	
	# print(spec) 
	# print(spec.shape)

	return spec



import wave
import contextlib

def find_video_length_2(fname):


	with contextlib.closing(wave.open(fname,'r')) as f:
	    frames = f.getnframes()
	    rate = f.getframerate()
	    duration = frames / float(rate)
	    return int(duration) - WINDOW_SLICE 


def preprocess_test(filepath):

	SUBJECT_LIST = [os.path.splitext(subject)[0] for subject in os.listdir(filepath)]
	print(SUBJECT_LIST)    
	print("yes")
	for subject in SUBJECT_LIST:
		
		feature_batch = []
		
		LENGTH_OF_VID = find_video_length_2( filepath+subject+'.wav' ) 
		print(LENGTH_OF_VID)
		for offset in range( 0 , LENGTH_OF_VID ):  # mapping it to the offsets so that it can be used later for generating the entire feature batch for the audio 
			feature_batch.append(generate_spec(filepath+"/"+subject+".wav",offset)) # 3600 corresponds to the 10 minute sample video we made for the testing 
			print("Now at offset " + str(offset) )



		feature_batch = np.asarray(feature_batch, dtype=np.float32)

		with open(TEST_FEATURE_BATCH_PATH + subject + '_feature_batch', 'wb') as fp:
			pickle.dump(feature_batch, fp)

			


preprocess_test(TEST_AUDIO_PATH)
