
import os 
import wave 
import cv2 
import numpy as np

audio_folder = 'test_audios'
video_folder = 'sed_folder' 
LIUM_FOLDER = 'LIUM_diarization_results'
AALTO_FOLDER = 'aalto_diarization_results'
DIAR_OUT_FOLDER = 'diarization_output_folder/'

from collections import defaultdict
aalto_speaker_labels = defaultdict(list)
Lium_speaker_labels = defaultdict(list)

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


	Lower_x_coordinate = int(width/2)
	Lower_y_coordinate = int(height) 
	Upper_x_coordinate = int(width)
	Upper_y_coordinate = int(height/1.35)

	out = cv2.VideoWriter(  DIAR_OUT_FOLDER + video_name, cv2.VideoWriter_fourcc('M','J','P','G'), fps , ( int(width),int(height) ) )
	frame_number = 0 
	Last_aalto_speaker = { } 
	Last_lium_speaker = {} 
	# Read until video is completed
	Total_detected_Speaker = 0 
	while(cap.isOpened()):
		  # Capture frame-by-frame
		ctr = ctr + 1
		ret, frame = cap.read()
		if ret == True:
			cv2.rectangle(frame,( Lower_x_coordinate, Lower_y_coordinate ),( Upper_x_coordinate, Upper_y_coordinate ),(0,0,0),-1)
			aalto_label_str = ""
			lium_label_str = ""
			if len(aalto_speaker_labels[frame_number]) == 1 :
				aalto_label_str = " Identified speaker: "

			if len(aalto_speaker_labels[frame_number]) > 1 :
				aalto_label_str = " Identified speakers: "	

			if len(Lium_speaker_labels[frame_number]) == 1 :
				lium_label_str = "  Lium identified speaker:  "

			if len(Lium_speaker_labels[frame_number]) > 1 :
				lium_label_str = "  Lium identified speakers: "	

			print(aalto_label_str)	
			cv2.putText(frame, aalto_label_str ,( Lower_x_coordinate	 , int(height-60) ),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255))
			aalto_label_str = ""
			lium_label_str = ""
			disp_ctr = 6
			for j in set(aalto_speaker_labels[frame_number]):
				disp_ctr = disp_ctr - 1 
				aalto_label_str = " speaker " + str(j).split('_')[1][0]  

				if j in Last_aalto_speaker:
					Last_time_stamp = Last_aalto_speaker[j]
					time_diff = np.ceil(  ( ctr - Last_time_stamp )/video_fps )
					seconds_start = int( Last_time_stamp/video_fps )  
					minutes_start = int( seconds_start/60 ) 
					seconds_start = int( seconds_start%60 )
					aalto_label_str = aalto_label_str + " : first spoke " + str(time_diff) + " sec ago at " + str(minutes_start)+":"+str(seconds_start) 	
				else :
					aalto_label_str = aalto_label_str + " : first time speaking "	
					Last_aalto_speaker[ j ] = ctr 					 
				
				
				print(aalto_label_str)

				cv2.putText(frame, aalto_label_str ,( Lower_x_coordinate	 , int(height-disp_ctr*10) ),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255))
				aalto_label_str =  ""
				print("what i want", j)

			

			cv2.putText(frame, " Total detected Speakers till now : %d "% len(Last_aalto_speaker) ,( Lower_x_coordinate	 , int(height-10) ),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255))
	
			# for j in set(Lium_speaker_labels[frame_number]):
			# 	lium_label_str = lium_label_str + " " + str(j)[1:] 
			# 	print("what i want", str(j)[1:])
					
			
			# cv2.putText(frame, aalto_label_str ,( Lower_x_coordinate	 , int(height-10) ),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255))
			# cv2.putText(frame, lium_label_str ,( Lower_x_coordinate	 , int(height-30) ),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255))				

			out.write(frame)
			# Display the resulting frame
			# cv2.imshow('Frame',frame)
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
	aalto_file_path = AALTO_FOLDER + '/' + audio_filename[:-4] + '_wav_stdout' 
	video_fps = get_video_fps(audio_filename)  
	Lium_speaker_labels.clear() 
	aalto_speaker_labels.clear()
	
	with open(aalto_file_path) as f:
		for line in f:
			print (line)
			segmented_info = line.split(' ')
			start_time = float(segmented_info[2].split('=')[1] )
			end_time = float(segmented_info[3].split('=')[1] )
			speaker = segmented_info[4].split('=')[1]
			print(start_time,end_time,speaker)
			start_video_frame = int( np.floor( start_time*video_fps ) )
			end_video_frame = int(np.ceil( end_time*video_fps ) )
			for idx in range ( start_video_frame,end_video_frame+1):
				aalto_speaker_labels[idx].append(speaker)


	lium_file_path  = LIUM_FOLDER + '/'+ cropped_audio_name + '/' + cropped_audio_name+'.d.3.seg'				
	with open(lium_file_path) as f:
		for line in f:
			# print (line[0])
			if(line[0] == ';'):
				continue 


			segmented_info = line.split(' ')
			start_time = float(segmented_info[2] ) / 100 
			end_time = float(segmented_info[3] ) / 100 + start_time 
			speaker = segmented_info[7]
			print("lium " ,start_time,end_time,speaker)
			start_video_frame = int( np.floor( start_time*video_fps ) )
			end_video_frame = int(np.ceil( end_time*video_fps ) )
			for idx in range ( start_video_frame,end_video_frame+1):
				Lium_speaker_labels[idx].append(speaker)

	video_generation( get_video_name(audio_filename) )		
	# break		



