#!/usr/bin/env python
# coding: utf-8

# In[1]:


from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from os import path
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import numpy


# In[1]:


def preprocess(filename):
    src = filename
    ext = src.split(".")[-1]
    """if(ext == "mp3"):
        dst = "sample.wav"

        # convert wav to mp3                                                            
        sound = AudioSegment.from_mp3(src)
        sound.export(dst, format="wav")
    else:
        os.rename(src,'sample.wav')
    """
    
    sound_file = AudioSegment.from_wav("sample.wav")
    audio_chunks = split_on_silence(sound_file, min_silence_len=500, silence_thresh=-16)

    for i, chunk in enumerate(audio_chunks):

        out_file = "chunk{0}.wav".format(i)
        print("exporting", out_file)
        chunk.export(out_file, format="wav")

    if(os.path.exists('chunk0.wav')):
        os.rename('chunk0.wav','sample.wav')

    (rate,sig) = wav.read('sample.wav')
    mfccss = mfcc(sig,samplerate=44100,winlen=0.02,winstep=0.01,numcep=39, nfilt=40,nfft=1024,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)


    # create a file to save our results in
    outputFile = "test.mfcc"
    file = open(outputFile, 'w+') # make file/over write existing file
    numpy.savetxt(file, mfccss, delimiter=" ") #save MFCCs as .csv
    file.close() # close file


# In[ ]:





# In[ ]:




