
import os 
import wave 
import cv2 
import numpy as np
import pickle

audio_folder = 'test_audios'
label_folder = 'prob_label_dump/'  
video_folder = 'diarization_output_folder/' 
output_folder = 'scene_output/'
from collections import defaultdict

scene_labels = defaultdict(list)

scene_decode = ['beach' , 'bus' ,  'public_place/meeting_room' , 'car' , 'public_place/meeting_room', 'forest_path',
 'public_place/meeting_room', 'home' ,'library', 'metro_station' , 'office', 'park',
 'residential_area', 'train' , 'tram']
print(scene_decode[2])


def get_video_name( audio_filename ):
    return audio_filename[:-4] + '.mp4'


def get_video_fps(audio_filename):
	video_fps = -1 
	cap = cv2.VideoCapture( video_folder +'/'+ get_video_name( audio_filename ) )
	# Check if camera opened successfully
	if (cap.isOpened()== False): 
	  print("Error opening video stream or file")
	video_fps =  np.ceil( cap.get(cv2.CAP_PROP_FPS) )
	return video_fps



def video_generation( video_name ):

	ctr = 1  
	# import skvideo.io
	cap = cv2.VideoCapture( video_folder +'/'+ get_video_name( audio_filename ) )
	# Check if camera opened successfully
	if (cap.isOpened()== False): 
	  print("Error opening video stream or file")
	# cv2.CAP_PROP_FRAME_WIDTH
	width =   cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
	height =  cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
	fps =  np.ceil( cap.get(cv2.CAP_PROP_FPS) )
	SCALE_FACTOR = 1

	print(width,height,fps)


	Lower_x_coordinate = int(width/2)
	Lower_y_coordinate = int(50) 
	Upper_x_coordinate = int(width)
	Upper_y_coordinate = int(0)

	out = cv2.VideoWriter( output_folder + video_name,cv2.VideoWriter_fourcc('M','J','P','G'), fps , ( int(width),int(height) ) )
	frame_number = 0 

	# Read until video is completed
	while(cap.isOpened()):
		  # Capture frame-by-frame
		ctr = ctr + 1
		ret, frame = cap.read()
		if ret == True:
			scene_label_str = "scene_labels: "
			scene_arr = [ ]
			if len(scene_labels[frame_number] ) < 2 :
				break
			for idx in range(0,2):
				scene_idx = scene_labels[frame_number][idx]
				scene_arr.append(  str( scene_decode[scene_idx] ) )  

			cv2.rectangle(frame,( Lower_x_coordinate, Lower_y_coordinate ),( Upper_x_coordinate, Upper_y_coordinate ),(0,0,0),-1)
			cv2.putText(frame, "top_2_possible_scenes" ,( Lower_x_coordinate , int(10) ),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255))			
			cv2.putText(frame, scene_arr[0] ,( Lower_x_coordinate , int(20) ),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255))
			cv2.putText(frame, scene_arr[1] ,( Lower_x_coordinate , int(30) ),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255))
			# cv2.putText(frame, scene_arr[2] ,( Lower_x_coordinate , int(40) ),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255))
			
			out.write(frame)
			# Display the resulting frame
			#cv2.imshow('Frame',frame)
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


	_audio_file = wave.open(audio_folder+'/'+audio_filename)
	# Audio info
	sample_rate = _audio_file.getframerate()
	cropped_audio_name = audio_filename[:-4]
	print(cropped_audio_name,sample_rate)
	video_fps = get_video_fps(audio_filename)  


	
	audio_label_arr = pickle.load( open( label_folder + cropped_audio_name +'_label' ,  "rb" ) )   
	print(len(audio_label_arr))
	scene_labels.clear()
	for i in range(0,len(audio_label_arr)):
		print(audio_label_arr) 
		start_time = i 
		end_time = i+1 
		start_video_frame = int( np.floor( start_time*video_fps ) )
		end_video_frame = int(np.ceil( end_time*video_fps ) )
		for idx in range ( start_video_frame,end_video_frame+1):
			scene_labels[idx].append(audio_label_arr[i][0])
			if scene_decode[ audio_label_arr[i][1] ] != scene_decode[ audio_label_arr[i][0] ] :
				scene_labels[idx].append(audio_label_arr[i][1])
				continue
			if scene_decode[ audio_label_arr[i][2] ] != scene_decode[ audio_label_arr[i][0] ] :
				scene_labels[idx].append(audio_label_arr[i][2])
				continue				

			if scene_decode[ audio_label_arr[i][3] ] != scene_decode[ audio_label_arr[i][0] ] :
				scene_labels[idx].append(audio_label_arr[i][3])
				continue		
			
			print(audio_label_arr[i][0:2])	
	video_generation(get_video_name( audio_filename ))		
	# break		



