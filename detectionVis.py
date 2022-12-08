import json
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from ast import literal_eval
import numpy as np

# enter paratmeters 
ingestion_src = './project-120722-1-ingestion.json'

ModelName = 'faster_rcnn_inception_resnet_v2'
# detections-resnet152
# resnet101
# ssd
# centernet
# x mask-rcnn
# resnet152
# resnet152(800x1333)
# x extremenet
# ssd_resnet152
# X effcientdet_d7
# faster_rcnn_inception_resnet_v2
# centernet_hg104

detection_src = './project-120722-1-detections-'+ ModelName +'.csv'

processed_video_src = 'dataset/processed1.avi'

f = open(ingestion_src)
data = json.load(f)
source = data['source-video']

df = pd.read_csv(detection_src,converters={"boxes": literal_eval})

frameNo = 1

df_currentFrame = df[df['frame']==frameNo]

def get_normal_coord(decimal_coords,im_height,im_width):
    ymin, xmin, ymax, xmax = decimal_coords
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    return [int(left),int(right),int(top),int(bottom)]


aim_score = 0.8

cap = cv2.VideoCapture(processed_video_src)
ret, frame = cap.read()
im_height,im_width,_ = frame.shape
out =cv2.VideoWriter('demo-1-('+ str(aim_score) + ')-'+ ModelName +'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (im_width,im_height))



cap = cv2.VideoCapture(processed_video_src)


frameNo = 0 

if (cap.isOpened()== False): 
    print("Error opening video stream or file")
else:       
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == True:
            pass
        else: 
            break
            
            
        im_height,im_width,_ = frame.shape
        image_np = frame.copy()
        
        frameNo = frameNo + 1 
        
        # Get detections here 
        df_currentFrame = df[df['frame']==frameNo]
        for ind,row in df_currentFrame.iterrows():
        # DETECTIONS 
            detections = row['boxes']
            score = row['scores']
            
            if score> aim_score:
            # plot boxes on 
                left, right, top, bottom = get_normal_coord(detections,im_height,im_width)
                image_np = cv2.rectangle(image_np, (int(left),int(top)), (int(right),int(bottom)), (0,0,255), 2)

        out.write(image_np)
        
out.release()
       