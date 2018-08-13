import os
import speech_recognition as sr
from tqdm import tqdm
from multiprocessing.dummy import Pool
pool = Pool(8) # Number of concurrent threads

with open("speech-to-text/api-key.json") as f:
	GOOGLE_CLOUD_SPEECH_CREDENTIALS = f.read()

r = sr.Recognizer()
files = sorted(os.listdir('speech-to-text/parts/'))

def transcribe(data):
	idx, file = data
	name = "speech-to-text/parts/" + file
	print(name + " started")
	# Load audio file
	text = ""
	with sr.AudioFile(name) as source:

		try:
			audio = r.record(source)
		except:
			print("hey")
		# Transcribe audio file
	

	
	try:
		text = r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
	except:
		print( "hey")

		
	print(name + " done")

	return {
		"idx": idx,
		"text": text
	}

all_text = pool.map(transcribe, enumerate(files))
pool.close()
pool.join()

transcript = ""
for t in sorted(all_text, key=lambda x: x['idx']):
	total_seconds = t['idx'] * 30
	# Cool shortcut from:
	# https://stackoverflow.com/questions/775049/python-time-seconds-to-hms
	# to get hours, minutes and seconds
	m, s = divmod(total_seconds, 60)
	h, m = divmod(m, 60)

	# Format time as h:m:s - 30 seconds of text
	transcript = transcript +  t['text'] + '\n'  

print(transcript)

with open("speech-to-text/transcript.txt", "w") as f:
	f.write(transcript)

