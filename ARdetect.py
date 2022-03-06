import numpy as np
import cv2
from matplotlib import pyplot as plt



video = "1tagvideo.mp4"


# create video capture object
video_ = cv2.VideoCapture(video)

# wait till video is playing
while not video_.isOpened():
    print("Loading Video")
    video_ = cv2.VideoCapture(video)
    cv2.waitKey(1000)

# read video frame by frame
pos_frame = video_.get(cv2.CAP_PROP_POS_FRAMES)
while True:
    flag, frame = video_.read()
    if flag:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pos_frame = video_.get(cv2.CAP_PROP_POS_FRAMES)
        print(str(pos_frame)+" frames")
        break    
    else:   # wait for next frame        
        video_.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
        cv2.waitKey(1000)
    if cv2.waitKey(10) == 27:
        break
    if video_.get(cv2.CAP_PROP_POS_FRAMES) == video_.get(cv2.CAP_PROP_FRAME_COUNT): 
        # traversed all frames
        break
# cv2.imwrite('frame.png', frame)

frame = cv2.GaussianBlur(frame, (7, 7), 0) #To obtain cleaner segmentation

# Convert image to bw by thresholding
(thresh, im_bw) = cv2.threshold(frame, 140, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
frame = im_bw



plt.figure(figsize=(10,10))
plt.imshow(frame)
plt.axis('off')
plt.show()