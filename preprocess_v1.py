# In[ ]:
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from python_speech_features import mfcc
import scipy.io.wavfile as wav
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import numpy


# In[1]:


def preprocess(filename):
    src = filename
    ext = src.split(".")[-1]
    
    if(ext == "mp3"):
        dst = "/tmp/sample.wav"
        # convert wav to mp3                                                            
        sound = AudioSegment.from_mp3(src)
        sound.export(dst, format="wav")
    if(src != "sample.wav"):
        loc = "/tmp/" + src
        os.rename(loc,'/tmp/sample.wav')
        
    sound_file = AudioSegment.from_wav("/tmp/sample.wav")
    audio_chunks = split_on_silence(sound_file, min_silence_len=200, silence_thresh=-16)

    for i, chunk in enumerate(audio_chunks):
        out_file = "/tmp/chunk{0}.wav".format(i)
        print("exporting", out_file)
        chunk.export(out_file, format="wav")

    if(os.path.exists('/tmp/chunk0.wav')):
        os.rename('/tmp/chunk0.wav','/tmp/data123.wav')
        (rate,sig) = wav.read('/tmp/data123.wav')
    else:
        (rate,sig) = wav.read('/tmp/sample.wav')
        
    mfccss = mfcc(sig,samplerate=44100,winlen=0.02,winstep=0.01,numcep=39, nfilt=40,nfft=1024,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)


    # create a file to save our results in
    outputFile = "/tmp/test.mfcc"
    file = open(outputFile, 'w+') # make file/over write existing file
    numpy.savetxt(file, mfccss, delimiter=" ") #save MFCCs as .csv
    file.close() # close file





# In[ ]:




