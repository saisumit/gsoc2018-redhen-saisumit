from __future__ import print_function
import wave
import numpy as np
import utils
import librosa
import os
from sklearn import preprocessing
import os
import numpy as np
import time
import sys
from sklearn.metrics import confusion_matrix
import metrics
import utils

import pandas as pd
import csv
import cv2
import subprocess 


MUTE_TEXT_FOLDER = 'Mute_Background/'
VIDEO_FOLDER = 'test_videos/'
MUTE_VIDEO_FOLDER  = 'mute_test_videos/'

def get_video_name( audio_name ):
	return audio_name[:-4]+'.mp4'

		
# ffmpeg -i video.mp4 -af "volume=enable='between(t,5,10)':volume=0, volume=enable='between(t,15,20)':volume=0"
for silent_txt in os.listdir(MUTE_TEXT_FOLDER):

	df = pd.read_csv(MUTE_TEXT_FOLDER + silent_txt)
	audio_name = silent_txt[:-4]+'.wav'
	command = "ffmpeg -i %s%s -af "% (VIDEO_FOLDER, get_video_name(audio_name ) ) 
	command = command + ' " ' 
	Last_idx = len(df)  
	for i in range( 0 , Last_idx ) :
		start_time = df.iloc[i][0]
		end_time = df.iloc[i][1] 
		command = command + "volume=enable='between(t,%f,%f)':volume=0"%(start_time,end_time)
		if i != Last_idx - 1:
			command = command + ", "



	command = command +' " '
	
	command = command + MUTE_VIDEO_FOLDER + silent_txt[:-4]+".mp4 "
	command = command + '\n'
	text_file = open("mute.sh", "a")
	text_file.write(command)
	text_file.close()

