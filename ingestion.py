import numpy as np
import cv2 as cv
import json 
import time

# Creates a JSON file with the bbox parameters marking entry and exit 

# set parameters
project = '120722-1-ingestion'
src = 'dataset/processed1.avi'

# Get a video Frame
cap = cv.VideoCapture(src)
nframe = 0
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
    print('Video DID NOT LOAD')
    quit()
else:
    while nframe < 50:
        ret, frame = cap.read()
        print('Read a new frame: ', ret)
        if ret:
            nframe = nframe + 1 
            pass
        else:
            cap.release()
            break


regions = []
drawing = False 
ix,iy = -1,-1
def draw_rectanlge(event, x, y, flags, param):
    global ix,iy,drawing,mode
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            cv.rectangle(img, (ix, iy), (x, y), (255, 255, 0), -1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        regions.append([ix,iy,x,y])



img = frame
cv.namedWindow('image') 
cv.setMouseCallback('image',draw_rectanlge)

while(1):
    cv.imshow('image',img)
    print(ix,iy)
    if cv.waitKey(20) & 0xFF == 27:
        break


# Data to be written 
dictionary ={ 
  "source-video": src, 
  "project": project, 
  "boxes": regions
} 
      
# Serializing json  
json_object = json.dumps(dictionary, indent = 4) 
with open(f"project-{project}.json", "w") as outfile:
    json.dump(dictionary, outfile)


cv.destroyAllWindows()