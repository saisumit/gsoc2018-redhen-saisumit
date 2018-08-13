
import os 
import wave 
import cv2 
import numpy as np

audio_folder = 'test_audios'
video_folder = 'scene_output' 
SPEECH_OUT_FOLDER = 'speech_output/'
speech_file = 'speech-to-text/transcript.txt'
from collections import defaultdict
speech_recog = defaultdict(list)
LENGTH = 5

def get_video_name( audio_filename ):
	return audio_filename[:-4] + '.mp4'


def get_video_fps(audio_filename):
	video_fps = -1 
	cap = cv2.VideoCapture( video_folder +'/'+ get_video_name(audio_filename) )
	# Check if camera opened successfully
	if (cap.isOpened()== False): 
	  print("Error opening video stream or file")
	video_fps =  np.ceil( cap.get(cv2.CAP_PROP_FPS) )
	return video_fps



def video_generation( video_name ):

	ctr = 1  
	# import skvideo.io
	cap = cv2.VideoCapture( video_folder +'/'+ get_video_name(audio_filename) )
	# Check if camera opened successfully
	if (cap.isOpened()== False): 
	  print("Error opening video stream or file")
	# cv2.CAP_PROP_FRAME_WIDTH
	width =   cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
	height =  cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
	fps =  np.ceil( cap.get(cv2.CAP_PROP_FPS) )
	SCALE_FACTOR = 1

	print(width,height,fps)


	Lower_x_coordinate = int(0)
	Lower_y_coordinate = int(height) 
	Upper_x_coordinate = int(width)
	Upper_y_coordinate = int(height )


	out = cv2.VideoWriter(  SPEECH_OUT_FOLDER + video_name, cv2.VideoWriter_fourcc('M','J','P','G'), fps , ( int(width),int(height+50) ) )
	frame_number = 0 

	while(cap.isOpened()):
		  # Capture frame-by-frame
		ctr = ctr + 1
		ret, frame = cap.read()
		if ret == True:

			new_frame =cv2.copyMakeBorder(frame,0,50,0,0,cv2.BORDER_CONSTANT,value=[0, 0,0])
			# cv2.rectangle(frame,( Lower_x_coordinate, Lower_y_coordinate ),( Upper_x_coordinate, Upper_y_coordinate ),(0,0,0),-1)
			text1 = "Recognized Speech: "  +  speech_recog[ctr][0][:70]
			text2 = '-'+speech_recog[ctr][0][70:]
			cv2.putText(new_frame, text1,( int( Lower_x_coordinate+ 10 )	 , int(height + 15 )),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255))
			cv2.putText(new_frame, text2,( int( Lower_x_coordinate+ 10 )	 , int(height + 25 )),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255))

			out.write(new_frame)
			# Display the resulting frame
			# cv2.imshow('Frame',new_frame)
			frame_number = frame_number + 1	
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

	
for audio_filename in os.listdir(audio_folder):
	
	video_fps = get_video_fps(audio_filename)  
	with open(speech_file) as f:
		ctr = 0  
		for line in f:
			print (line)
			start_time = LENGTH*ctr  
			end_time =  start_time + LENGTH 
			ctr = ctr +  1 
			print(start_time,end_time)
			start_video_frame = int( np.floor( start_time*video_fps ) )
			end_video_frame = int(np.ceil( end_time*video_fps ) )
			for idx in range ( start_video_frame,end_video_frame+1):
				speech_recog[idx].append(str(line))

	video_generation( get_video_name(audio_filename) )		
	# break		



