

import cv2
import numpy as np
from tqdm import tqdm
import os

import tensorflow as tf
import pathlib
from PIL import Image
import pandas as pd
import time
import pybboxes as pbx
from matplotlib import pyplot as plt



# Set paramaters
src = 'dataset/raw1.mp4'
write_path = 'dataset/processed1.avi'
skip = 3
finalFrame = 4500




cap = cv2.VideoCapture(src)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# output
out =cv2.VideoWriter(write_path,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


print('length : ', length)
skip = int(input('Enter the skip length\n'))
finalFrame = int(input('Enter the final frame\n'))



assert length > finalFrame


# To shorten + compress a video
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
i = 0 

for i in tqdm(range(finalFrame)):
    
    ret, frame = cap.read()
    if i%skip == 0 :
        out.write(frame)
        
    if ret == True:
#         cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else: 
        break
        
cap.release()
out.release()
# cv2.destroyAllWindows()