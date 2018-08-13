

# takes the text files from the Mute Background
# takes the video from test_videos/ 
# puts muted video in mute_test_video folder


 ffmpeg -i mute_test_videos/input.mp4 -vn mute_test_videos/input.wav 


 rm -r lium-diarization/test_wav    #removes the stored the videos in the lium_diarization if any 
 mkdir lium-diarization/test_wav 


 mv mute_test_videos/input.wav lium-diarization/test_wav/input.wav  
 cd lium-diarization
 ./go.sh  #this runs the lium diarization toolkit over the muted audios to do speaker diarization
 cd ../
 rm -r LIUM_diarization_results  #this removes the result folder from the lium diarization 
 mv  lium-diarization/test_out LIUM_diarization_results #this moves the result of diarization toolkit from the test_out to LIUM_diarization results to be used by final module 



# #run the aalto diarization using docker 
 rm -r mono_sample_folder 
 mkdir mono_sample_folder 

 rm -r aalto_diarization_results
 mkdir aalto_diarization_results

 ffmpeg -i test_audios/input.wav -ac 1 -ar 16000 mono_sample_folder/input.wav


 docker kill aalto_spch_diarization
 docker rm aalto_spch_diarization
 docker create -t -i --name aalto_spch_diarization blabbertabber/aalto-speech-diarizer bash
 docker start aalto_spch_diarization
 docker cp   mono_sample_folder/input.wav aalto_spch_diarization:speaker-diarization/
 docker exec -ti aalto_spch_diarization sh -c "cd speaker-diarization && ./spk-diarization2.py input.wav"
 docker cp aalto_spch_diarization:speaker-diarization/stdout aalto_diarization_results/
 mv aalto_diarization_results/stdout aalto_diarization_results/input_wav_stdout



 docker kill aalto_spch_diarization
 docker rm aalto_spch_diarization
#takes normal audio from test audio and send it to mono_sample_folder with mono channel and 16000 
#takes sampled audio from mono folder runs aalto.
#puts result in aalto diarization results 
#close docker container 

 rm -r diarization_output_folder
 mkdir diarization_output_folder
 python speaker_diarization.py

# #stores the diarized output videos in diarization_output_folder


 rm -r test_feature_batch 
 mkdir test_feature_batch 
 python scene_feature_gen.py
# takes audio from test_audios
# takes model from fin model
# dumps feature batch in test_feature_batch


 rm -r prob_label_dump
 mkdir prob_label_dump
 python scene_label_generator.py

#takes feature from test_feature batch 
#dumps labels and probability in prob_label_dump

 rm -r scene_output
 mkdir scene_output 
 python scene.py
#takes video from diarization output 
#takes label / features from prob_label_dump 
#puts output in scene_output

 rm -r final_output
 mkdir final_output

 ffmpeg -i scene_output/input.mp4 -i test_audios/input.wav -c:v copy -c:a aac -strict experimental final_output/result.mp4

#store the corresponding files in the result folder 


#cleaning up 
 rm -r test_videos/  # Removing the test videos folder
 rm -r test_audios/  # remove the test audio folder 
