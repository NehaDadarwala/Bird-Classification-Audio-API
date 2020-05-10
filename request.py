# -*- coding: utf-8 -*-
"""
Created on Thu May  7 23:08:28 2020

@author: Neha Dadarwala
"""

import requests

url = 'http://localhost:5000/predict'
files = {'file': open('sample.wav', 'rb')}
r = requests.post(url, files=files)

print(r.json())