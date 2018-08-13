
import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix 
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
import pickle
import copy 


labelencoder = LabelEncoder() 
onehot_encoder = OneHotEncoder(sparse=False)
TEST_FEATURE_BATCH_PATH = 'test_feature_batch/'
TEST_FILE_PATH = 'test_audios/'
N_TRAIN_STEPS = 5
NUM_OF_BANDS = 200
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MODEL_NAME = 'sai_audio.tfl'
TEST_DUMP_FOLDER = 'prob_label_dump/'
TENSORBOARD_PATH = 'tensor_logs/'
# train = pd.read_csv('../train.csv') 
# scene_list = train['Scene'].unique()
# print(scene_list)
# train['Scene']=labelencoder.fit_transform(train['Scene']);

# print(labelencoder.transform(scene_list))


def get_video_name( audio_filename ):
    return audio_filename[:-4] + '.mp4'


 
def one_hot_encode(label_batch):
		label_batch = np.asarray(label_batch)
		integer_encoded = label_batch.reshape(len(label_batch), 1)
		onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
		label_batch = onehot_encoded
		ans = np.unique(label_batch)
		for i in ans:
			print(i)
		return label_batch

def sai_net( ):

	network = input_data(shape=[None, NUM_OF_BANDS, 500, 1 ], name='features')
	print(network.shape) 
	network = conv_2d(network, 100 , [NUM_OF_BANDS,50 ] , strides = [NUM_OF_BANDS,1] )
	print(network.shape) 
	network = tflearn.layers.batch_normalization(network)
	print(network.shape ) 
	network = tflearn.activations.relu(network)
	print(network.shape ) 
	network = dropout(network, 0.25)
	print(network.shape ) 
	network = conv_2d(network, 100 , [1,1])
	print(network.shape ) 
	network = tflearn.layers.batch_normalization(network)
	network = tflearn.activations.relu(network)
	network = dropout(network, 0.25)
	network = conv_2d(network, 15, [1,1], activation='softmax')
	print(network.shape ) 
	network = tflearn.layers.conv.global_avg_pool (network, name='GlobalAvgPool')
	print(network.shape)
	network = regression(network, optimizer='momentum', loss='categorical_crossentropy', 
						 learning_rate=LEARNING_RATE, name='labels')

	model = tflearn.DNN( network,tensorboard_dir=TENSORBOARD_PATH, tensorboard_verbose=3 )
	return model






def predict_model():

  	model= sai_net()
  	model.load('fin_model/sai_audio.tfl')
  	SUBJECT_LIST = [os.path.splitext(subject)[0] for subject in os.listdir(TEST_FILE_PATH)]

	for subject in SUBJECT_LIST:
	  	features = [] 
	  	labels = []
		print(subject)
		feature_batch = pickle.load( open(TEST_FEATURE_BATCH_PATH + subject +'_feature_batch' ,  "rb" ) )   
		features.extend(feature_batch) 

		features = np.array(features)
		features = features.reshape([-1, NUM_OF_BANDS, 500 , 1]) 
	

		y_predict_prob = model.predict(features)
		y_predict_label = model.predict_label({'features':features})  

	
		print(y_predict_prob)
		print(len(y_predict_label))
		print(y_predict_label)
		with open( TEST_DUMP_FOLDER+subject+'_label', 'wb') as fp:
			pickle.dump( y_predict_label , fp) 
		with open( TEST_DUMP_FOLDER+subject+'_prob', 'wb') as fp:
			pickle.dump( y_predict_prob , fp) 		      

	

predict_model() 
