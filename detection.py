import json
import cv2
from tqdm import tqdm
import tensorflow as tf
import pathlib
from PIL import Image
import pandas as pd
import numpy as np
import time
import pybboxes as pbx
from matplotlib import pyplot as plt



# Parameters To Edit
ingestion_src = './project-120722-1-ingestion.json' 

model_prefix = 'faster_rcnn_inception_resnet_v2'

# detections-resnet152
# resnet101
# ssd
# centernet
# mask-rcnn (failed)
# resnet152
# resnet152(800x1333)
# extremenet (failed)
# ssd_resnet152
# effcientdet_d7
# faster_rcnn_inception_resnet_v2
# centernet_hg104




# Model Definition

ModelPath = 'faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8'

# faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8
# faster_rcnn_resnet101_coco_2018_01_28
# ssd_mobilenet_v1_coco_2018_01_28
# centernet_hg104_512x512_coco17_tpu-8
# mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8
# faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8
# extremenet
# ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8
# efficientdet_d7_coco17_tpu-32
# faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8
# centernet_hg104_1024x1024_kpts_coco17_tpu-32

model_dir = './models/' + ModelPath +'/saved_model/'


model = tf.saved_model.load(str(model_dir))
model_fn = model.signatures['serving_default']



def detectPersons(image_np,model_fn):
    start = time.time()
    image = np.asarray(image_np)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]
    output_dict = model_fn(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    op = {
        'classes':output_dict['detection_classes'].tolist(),
        'boxes':output_dict['detection_boxes'].tolist(),
        'scores':output_dict['detection_scores'].tolist()
    }
    op = pd.DataFrame(op)
    op = op[op['classes']==1]
    end = time.time()
    return op,end-start




    

f = open(ingestion_src)
data = json.load(f)
source = data['source-video']

cap = cv2.VideoCapture(source)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


OP = []
frameNo = 0 
fullTime = 0


if (cap.isOpened()== False): 
    print("Error opening video stream or file")
else:       
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == True:
            pass
        else: 
            break
        image_np = frame.copy()
        
        frameNo = frameNo + 1 
        detections, time_taken = detectPersons(image_np,model_fn)
        detections['frame'] = frameNo
        OP.append(detections)

        print(f'Running Detection: {frameNo}/{length},Time: {time_taken}')
        fullTime = time_taken + fullTime



print(f'Avg Time = {fullTime/frameNo}')

       

OUTPUT = pd.concat(OP)
csvName = ingestion_src.split('.json')[-2].split('/')[-1].split('ingestion')[0] + 'detections' +'-' + model_prefix+ '.csv'
OUTPUT.to_csv(csvName)