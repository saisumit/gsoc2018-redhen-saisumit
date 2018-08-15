# gsoc2018-redhen-saisumit

Mentors:  
* Mehul Bhatt ( http://www.mehulbhatt.org/ ) 
* Jakob Suchan ( https://cosy.informatik.uni-bremen.de/staff/jakob-suchan.html )   
* Sri Krishna ( https://skrish13.github.io ) 
 
Organisation : Redhen Labs ( http://www.redhenlab.org/ )


# Audio Analysis of Egocentric and Third-person videos using Red-Hen Labs : #


The following library provides the functionality of the following types:  

1. Speech Identification ( When the person is speaking or not ) 
2. Speaker Diarization using Lium and Aalto diarization libraries. 
3. Scene Identification 
4. Speech Recognition 

# Running the pipeline #
1. Clone the repository 
2. cd Audio-Analysis-RedHen-saisumit
3. Place the desired input file as input.mp4. 
4. Run one of the followig undermentioned pipelines. 

  
# Speech Identification #

Speech identification uses the problem of identifying the speaking regions in the media. It classifies the regions in the 3 categories : 

* Silent Regions 
* People-speaking Regions
* Other Regions   


```

 ./audio_runner.sh -f sound_event_detection

```
References :  

* http://dcase.community/challenge2018/index
* http://www.cs.tut.fi/sgn/arg/dcase2017/documents/challenge_technical_reports/DCASE2017_Adavanne_130.pdf


# Speaker Diarization #
Speaker diarization is a task of identifying the speaker and indexing those speakers. Support of two diarizartion libraries is provided in this project : 

* Aalto Diarization Tool 
* Lium Diarization Tool 

Aalto is used for displaying the video results but Lium Results are also avaiable in text format which can be accessed as described in the Results. 

```
 ./audio_runner.sh  -f speaker_diarization
```

References : 

* http://liumtools.univ-lemans.fr
* https://github.com/ahmetaa/lium-diarization
* https://github.com/aalto-speech/speaker-diarization

 
# Scene Identification #
Scene identification is task of identifying the scene of the media. Following categories are considered and classification is provided based on that: 

* Bus
* Car
* City Center 
* Residential Area/Meeting Roomn 
* Home 
* Beach
* Library 
* Metro Station 
* Office
* Train 
* Tram 
* Park 
* Pub 


```
 ./audio_runner.sh  -f scene_identification

```

References: 

* https://github.com/karoldvl/paper-2017-DCASE
* http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/

# Speech Recognition  #

Speech Recognition is identifying what the person is speaking. It makes use of Google Cloud Speech-to-text 
to identify what the person is speaking. It runs the complete pipeline giving you the combined results of all the previous pipelines. 
 
References: 

* https://cloud.google.com/speech-to-text/
* https://www.alexkras.com/transcribing-audio-file-to-text-with-google-cloud-speech-api-and-python/
 

```
 ./audio_runner.sh 

```

## RESULTS ##

There are two types of results available with this pipeline.

1. The final video output with results are available in final_output/result.mp4
2. Also all the intermediate results of individual pipeline are available in Output_Files/
