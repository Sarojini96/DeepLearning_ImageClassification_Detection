#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:29:32 2020

@author: sarojini
"""


import imutils
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image,ImageDraw
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array

# load the image    
image = cv2.imread("rhs2.png")
orig = image.copy()
# pre-process the image for classification
image = cv2.resize(image, (64, 64))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
#model = load_model(args["model"])
classifier = load_model('test_model1.h5')

# classify the input image
array = classifier.predict(image)[0]
print("Result = ",array)
result = array[0]
answer = np.argmax(result)
print(answer)
label = "LHS" if array <0.5 else "RHS"
 
'''
# build the label
label = "lhs" if result >=0.34 else "rhs"
out = label , result[0]
print(out)
print(label)'''
#proba = result if result > 0.5 else "anticlockwise"
#label = "{}: {:.2f}%".format(label, proba * 100)
# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label ,(10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, ( 255,0,0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
