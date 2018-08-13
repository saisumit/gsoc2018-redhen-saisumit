

from __future__ import print_function
import wave
import numpy as np
import utils
import librosa
# from IPython import embed
import os
from sklearn import preprocessing
import os
import numpy as np
import time
import sys
import matplotlib.pyplot as plot
from keras.layers import Bidirectional, TimeDistributed, Conv2D, MaxPooling2D, Input, GRU, Dense, Activation, Dropout, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from sklearn.metrics import confusion_matrix
import metrics
import utils
# from IPython import embed
import keras.backend as K
import pandas as pd
import csv
import cv2
# KERAS_BACKEND=tensorflow python -c "from keras import backend"

K.set_image_data_format('channels_first')
plot.switch_backend('agg')
sys.setrecursionlimit(10000)
SILENT_FOLDER = 'Mute_Background/'
WEIGHTS_PATH = 'models/mon_2018_05_26_05_07_58_fold_4_model.h5'
SED_LABEL_FOLDER = 'sed_folder/'

# def load_data(_feat_folder, _mono, _fold=None):
#     feat_file_fold = os.path.join(_feat_folder, 'mbe_{}_fold{}.npz'.format('mon' if _mono else 'bin',))
#     dmp = np.load(feat_file_fold)
#     _X_train, _Y_train = dmp['arr_0'],  dmp['arr_1']
#     return _X_train, _Y_train 
sed_intervals = [] 
from collections import defaultdict
scene_labels = defaultdict(list)
# scene_labels = { } 
# for i in range(0,6):
#     sed_intervals[i] = 


inverse_class_labels = {
	0:'brakes squeaking',
	1:'car',
	2:'children',
	3:'large vehicle',
	4:'people speaking',
	5:'people walking'
}



def video_generation( video_name ):

	silence_list = [] 
	ctr = 1  
	# import skvideo.io
	cap = cv2.VideoCapture('test_videos/'+ video_name )
	# Check if camera opened successfully
	if (cap.isOpened()== False): 
	  print("Error opening video stream or file")
	# cv2.CAP_PROP_FRAME_WIDTH
	width =   cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
	height =  cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
	fps =  np.ceil( cap.get(cv2.CAP_PROP_FPS) )
	SCALE_FACTOR = 1
	width = width*SCALE_FACTOR 
	height = height*SCALE_FACTOR 
	print(width,height,fps)
	print_width = int(width/2)
	print_height  = int( height/1.15 )

	out = cv2.VideoWriter(SED_LABEL_FOLDER+video_name,cv2.VideoWriter_fourcc('M','J','P','G'), fps , ( int(width),int(height) ) )
	frame_number = 0 

	# Read until video is completed
	while(cap.isOpened()):
	  # Capture frame-by-frame
	  ctr = ctr + 1
	  ret, frame = cap.read()
	  if ret == True:
		
		frame = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR) 
		print(scene_labels[frame_number])
		label_str = ""
		speak_flag = False 
		other_flag = False
		silent_flag = True 

		for j in scene_labels[frame_number]:
			if j == 4:
				speak_flag = True 
				silent_flag = False
				other_flag = False 
				break
			else: 
				other_flag = True
				silent_flag = False 


		if speak_flag == True:
			label_str = "People Speaking"
			silence_list.append(0) 
		if silent_flag == True : 
			label_str = "Silence"
			silence_list.append(1 ) 
		if other_flag == True :
			label_str = "Other"    
			silence_list.append(1)     
	  
		cv2.rectangle(frame,(0,int(height)),(print_width,print_height),(0,0,0),-1)
		frame_number = frame_number + 1
		cv2.putText(frame, label_str ,( int(print_width/4) , int(height-10) ),cv2.FONT_HERSHEY_COMPLEX_SMALL,.4,(225,255,255))
		out.write(frame)
		# Display the resulting frame
		#cv2.imshow('Frame',frame)
	 
		# Press Q on keyboard to  exit
		if cv2.waitKey(1) & 0xFF == ord('q'):
		  break
	 
	  # Break the loop
	  else: 
		break
	# When everything done, release the video capture object
	cap.release()
	out.release()
	# Closes all the frames
	cv2.destroyAllWindows()

	Last_idx = len(silence_list)
	silence_start_frame = -1  
	silence_end_frame = -1  
	SILENCE_FLAG = False 
	silence_time_window = [ ]

	for i in range( 0 , Last_idx ): 
		
		if SILENCE_FLAG == False and silence_list[i] == 0 : 
			continue 

		if SILENCE_FLAG == True and i == Last_idx - 1 :
			silence_time_window.append( ( silence_start_frame/fps, i/fps ) )
			continue
			
			
		if silence_list[i] == 0:
			SILENCE_FLAG = False 
			silence_time_window.append( ( silence_start_frame/fps , silence_end_frame/fps  ) )
			silence_start_frame = -1 
			silence_end_frame = -1  
			continue


		if SILENCE_FLAG == True and silence_list[i] == 1 :
			silence_end_frame = i 

		if SILENCE_FLAG == False and silence_list[i] == 1 :
			silence_start_frame = i 
			silence_end_frame = i
			SILENCE_FLAG  = True


	with open( SILENT_FOLDER + video_name[:-4]+ '.txt' ,'wb') as out:
		csv_out=csv.writer(out)
		csv_out.writerow(['start_time','end_time'])
		for row in silence_time_window:
			csv_out.writerow(row)




			

def get_model(data_in, _cnn_nb_filt, _cnn_pool_size, _rnn_nb, _fc_nb):

	print("this is imp_stuff",data_in.shape[-3], data_in.shape[-2], data_in.shape[-1])

	spec_start = Input(shape=(data_in.shape[-3], data_in.shape[-2], data_in.shape[-1]))
	spec_x = spec_start
	for _i, _cnt in enumerate(_cnn_pool_size):
		spec_x = Conv2D(filters=_cnn_nb_filt, kernel_size=(3, 3), padding='same')(spec_x)
		spec_x = BatchNormalization(axis=1)(spec_x)
		spec_x = Activation('relu')(spec_x)
		spec_x = MaxPooling2D(pool_size=(1, _cnn_pool_size[_i]))(spec_x)
		spec_x = Dropout(dropout_rate)(spec_x)
	spec_x = Permute((2, 1, 3))(spec_x)
	spec_x = Reshape((data_in.shape[-2], -1))(spec_x)

	for _r in _rnn_nb:
		spec_x = Bidirectional(
			GRU(_r, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
			merge_mode='mul')(spec_x)

	for _f in _fc_nb:
		spec_x = TimeDistributed(Dense(_f))(spec_x)
		spec_x = Dropout(dropout_rate)(spec_x)

	spec_x = TimeDistributed(Dense(6))(spec_x)
	out = Activation('sigmoid', name='strong_out')(spec_x)

	_model = Model(inputs=spec_start, outputs=out)
	_model.compile(optimizer='Adam', loss='binary_crossentropy')
	_model.summary()
	return _model



def preprocess_data(_X, _Y,_seq_len, _nb_ch):
	# split into sequences
	_X = utils.split_in_seqs(_X, int(_seq_len) )
	_Y = utils.split_in_seqs(_Y, int(_seq_len) )

	_X = utils.split_multi_channels(_X, _nb_ch)

	return _X, _Y


def load_model(data_in, _cnn_nb_filt, _cnn_pool_size, _rnn_nb, _fc_nb,weights_path):
   model = get_model(data_in, _cnn_nb_filt, _cnn_pool_size, _rnn_nb, _fc_nb)
   model.load_weights(weights_path)
   return model


def get_video_name( audio_filename ):
	return audio_filename[:-4] + '.mp4'


def load_audio(filename, mono=True, fs=44100):
	"""Load audio file into numpy array
	Supports 24-bit wav-format
	
	Taken from TUT-SED system: https://github.com/TUT-ARG/DCASE2016-baseline-system-python
	
	Parameters
	----------
	filename:  str
		Path to audio file

	mono : bool
		In case of multi-channel audio, channels are averaged into single channel.
		(Default value=True)

	fs : int > 0 [scalar]
		Target sample rate, if input audio does not fulfil this, audio is resampled.
		(Default value=44100)

	Returns
	-------
	audio_data : numpy.ndarray [shape=(signal_length, channel)]
		Audio

	sample_rate : integer
		Sample rate

	"""

	file_base, file_extension = os.path.splitext(filename)
	if file_extension == '.wav':

		_audio_file = wave.open(filename)

		# Audio info
		sample_rate = _audio_file.getframerate()
		sample_width = _audio_file.getsampwidth()
		number_of_channels = _audio_file.getnchannels()
		number_of_frames = _audio_file.getnframes()
		print("info ",sample_rate,sample_width,number_of_channels,number_of_frames)

		# Read raw bytes
		data = _audio_file.readframes(number_of_frames)
		_audio_file.close()

		# Convert bytes based on sample_width
		num_samples, remainder = divmod(len(data), sample_width * number_of_channels)
		if remainder > 0:
			raise ValueError('The length of data is not a multiple of sample size * number of channels.')
		if sample_width > 4:
			raise ValueError('Sample size cannot be bigger than 4 bytes.')

		if sample_width == 3:
			# 24 bit audio
			a = np.empty((num_samples, number_of_channels, 4), dtype=np.uint8)
			raw_bytes = np.fromstring(data, dtype=np.uint8)
			a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)
			a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
			audio_data = a.view('<i4').reshape(a.shape[:-1]).T
		else:
			# 8 bit samples are stored as unsigned ints; others as signed ints.
			dt_char = 'u' if sample_width == 1 else 'i'
			a = np.fromstring(data, dtype='<%s%d' % (dt_char, sample_width))
			audio_data = a.reshape(-1, number_of_channels).T

		if mono:
			# Down-mix audio
			audio_data = np.mean(audio_data, axis=0)

		# Convert int values into float
		audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

		# Resample
		if fs != sample_rate:
			audio_data = librosa.core.resample(audio_data, sample_rate, fs)
			sample_rate = fs

		return audio_data, sample_rate
	return None, None




def extract_mbe(_y, _sr, _nfft, _nb_mel):
	# spec, n_fft = librosa.core.spectrum._spectrogram(y=_y, n_fft=_nfft, hop_length=, power=2)
	# mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel)
	# print(mel_basis,spec)
	# return np.log(np.dot(mel_basis, spec))
	spec = librosa.feature.melspectrogram(_y, sr=_sr, n_fft= nfft, hop_length= 1024,
											  n_mels=_nb_mel, fmax=22050, power=int(1))
	spec = librosa.power_to_db(spec)
	return spec 



# ###################################################################
#              Main script starts here
# ###################################################################

is_mono = True 

__class_labels = {
	'brakes squeaking': 0,
	'car': 1,
	'children': 2,
	'large vehicle': 3,
	'people speaking': 4,
	'people walking': 5
}

# location of data.
folds_list = [1, 2, 3, 4]
evaluation_setup_folder = 'evaluation_setup'
audio_folder = 'test_audios/'

# Output
feat_folder = 'test_feat/'
utils.create_folder(feat_folder)

# User set parameters
nfft = 2048
win_len = nfft
hop_len = 1024
nb_mel_bands = 40
sr = 44100

# -----------------------------------------------------------------------
# Feature extraction and label generation
# -----------------------------------------------------------------------
# Load labels
# train_file = os.path.join(evaluation_setup_folder, 'street_fold{}_train.txt'.format(1))
# evaluate_file = os.path.join(evaluation_setup_folder, 'street_fold{}_evaluate.txt'.format(1))

# Extract features for all audio files, and save it along with labels
for audio_filename in os.listdir(audio_folder):
	print (audio_filename)

for audio_filename in os.listdir(audio_folder):
	# break
	# audio_filename = 'driving_fails.wav'
	print(audio_filename[:-4])
	audio_file = os.path.join(audio_folder, audio_filename)
	print('Extracting features and label for : {}'.format(audio_file))
	y, sr = load_audio(audio_file, mono=is_mono, fs=sr)
	mbe = None

	if is_mono:
		mbe = extract_mbe(y, sr, nfft, nb_mel_bands).T
	else:
		for ch in range(y.shape[0]):
			mbe_ch = extract_mbe(y[ch, :], sr, nfft, nb_mel_bands).T
			if mbe is None:
				mbe = mbe_ch
			else:
				mbe = np.concatenate((mbe, mbe_ch), 1)

	label = np.zeros((mbe.shape[0], len(__class_labels)))
	# tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(audio_filename, 'mon' if is_mono else 'bin'))
	# np.savez(tmp_feat_file, mbe, label)







	X_test, Y_test = None, None

	# tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(audio_filename, 'mon' if is_mono else 'bin'))
	# dmp = np.load(tmp_feat_file)
	# tmp_mbe, tmp_label = dmp['arr_0'], dmp['arr_1']
	# print(len(tmp_mbe),len(tmp_label))
	# print(len(tmp_mbe[0]))

	# print(len(tmp_label[0]))
	if X_test is None:
		X_test, Y_test = mbe, label
	else:
		X_test, Y_test = np.concatenate((X_test, mbe), 0), np.concatenate((Y_test, label), 0)

	# Normalize the training data, and scale the testing data using the training data weights

	scaler = preprocessing.StandardScaler()
	X_test = scaler.fit_transform(X_test)

	# normalized_feat_file = os.path.join(feat_folder, 'mbe_{}_fold{}.npz'.format('mon' if is_mono else 'bin', 1))
	# np.savez(normalized_feat_file, X_test, Y_test)
	# print('normalized_feat_file : {}'.format(normalized_feat_file))



	is_mono = True  # True: mono-channel input, False: binaural input

	# feat_folder = 'feat/'

	nb_ch = 1 if is_mono else 2
	batch_size = 128    # Decrease this if you want to run on smaller GPU's
	seq_len = 256       # Frame sequence length. Input to the CRNN.
	nb_epoch = 50      # Training epochs
	patience = int(0.25 * nb_epoch)  # Patience for early stopping

	# Number of frames in 1 second, required to calculate F and ER for 1 sec segments.
	# Make sure the nfft and sr are the same as in feature.py
	sr = 44100
	nfft = 2048



	# CRNN model definition
	cnn_nb_filt = 128            # CNN filter size
	cnn_pool_size = [5, 2, 2]   # Maxpooling across frequency. Length of cnn_pool_size =  number of CNN layers
	rnn_nb = [32, 32]           # Number of RNN nodes.  Length of rnn_nb =  number of RNN layers
	fc_nb = [32]                # Number of FC nodes.  Length of fc_nb =  number of FC layers
	dropout_rate = 0.5          # Dropout after each layer
	print('MODEL PARAMETERS:\n cnn_nb_filt: {}, cnn_pool_size: {}, rnn_nb: {}, fc_nb: {}, dropout_rate: {}'.format(
		cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate))



	X, Y = preprocess_data(X_test, Y_test, seq_len, nb_ch)

	model = load_model(X, cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb,WEIGHTS_PATH)


	pred = model.predict(X)
	# print(pred)

	sum =  len(pred)*256 
	print(sum)
	for i in range(0,len(pred)):
		# print(len(pred))
		for j in range(0,len(pred[i])):
			print(pred[i][j])
			for k in range(0,len(pred[i][j])):

				if( k == 5 ):
					if(pred[i][j][k] > 0.85 ):
						pred[i][j][k] = 1 
						continue 
				else :           
					if( pred[i][j][k] >= 0.45 ):
						pred[i][j][k] = 1 
						continue
				
				pred[i][j][k] = 0 



	# print(pred)
	print(sum)

	video_fps = 0 
	cap = cv2.VideoCapture('test_videos/'+ get_video_name( audio_filename ) )
	# Check if camera opened successfully
	if (cap.isOpened()== False): 
	  print("Error opening video stream or file")
	video_fps =  np.ceil( cap.get(cv2.CAP_PROP_FPS) )
	# When everything done, release the video capture object
	cap.release()
	# Closes all the frames
	cv2.destroyAllWindows()



	for k in range(0,6):
		start_frame = end_frame = -1
		flag = False 

		pred_length = len(pred) - 1 
		pred_i_length = len(pred[pred_length]) - 1  

		for i in range(0,len(pred)):
			for j in range(0,len(pred[i])):

  
				frame_number =  i*256+ j
				frame_time = (frame_number*hop_len ) / sr

				if( pred[i][j][k] == 1 and (not( i== pred_length and j == pred_i_length )) ):
					if ( flag == True ):
						end_frame = max(end_frame,frame_number) 
					else:
						flag = True
						start_frame = end_frame = frame_number 

				else:

					if( flag == True):
   
						start_time =  float( start_frame*hop_len ) / sr                        
						end_time =  float( end_frame*hop_len ) / sr
						start_video_frame = int( np.floor( start_frame*hop_len*video_fps/ sr ) )
						end_video_frame = int(np.ceil(end_frame*hop_len*video_fps/sr ) )
						print("start_video_frame", start_video_frame,end_video_frame)
						print(k)
						print(inverse_class_labels)
						for idx in range ( start_video_frame,end_video_frame+1):
							scene_labels[idx].append(k)

						print(start_time,end_time)
						print(start_frame,end_frame)
						sed_intervals.append( ( start_time,end_time,k,audio_filename) )

					flag = False 
					start_time = end_time = -1

			  
	video_generation( get_video_name(audio_filename ) )
	scene_labels.clear()
	print("video_fps ",video_fps) 
	# break



	# for i in range ( 0 , 6 ):
	#     print("change ",inverse_class_labels[i])

	#     for j in range(0,len(sed_intervals[i])):
	#         print(sed_intervals[i][j])

	# # # print(sed_intervals.values())        
	# with open("test.csv", "wb") as outfile:
	#    writer = csv.writer(outfile)
	#    writer.writerow(sed_intervals.keys())
	#    writer.writerows(zip(sed_intervals.values()))



# print(sed_intervals)

sed_speaking_intervals = [ ]
for i in sed_intervals:
	print( i ) 
	if i[2] != 4:
		continue 
	else:
		tup = (i[0],i[1],"People_Speaking",i[3])
		sed_speaking_intervals.append( tup ) 

print(sed_speaking_intervals)

		

with open('people_speaking.csv','wb') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['start_time','end_time','sound_event','audio_id'])
    for row in sed_speaking_intervals:
        csv_out.writerow(row)



