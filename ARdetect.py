import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.fftpack as fp



video = "1tagvideo.mp4"


# create video capture object
video_ = cv2.VideoCapture(video)

def EdgeDetect(frame):
    frame = cv2.GaussianBlur(frame, (7, 7), 0) #To obtain cleaner segmentation
    # Convert image to bw by thresholding
    #cv2.threshold(src, thresholdValue, maxVal, thresholdingTechnique)
    (thresh, im_bw) = cv2.threshold(frame, 140, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    frame = im_bw
    
    cv2.imwrite('BeforeFFT.png',frame)
    
    # FFT
    F1 = fp.fft2((frame).astype(float))
    F2 = fp.fftshift(F1)
    
    # Create High pass filter
    (frame_width, frame_height) = frame.shape
    print(frame_width,' ',frame_height)
    mid_width, mid_height = int(frame_width/2), int(frame_height/2)
    n = 25
    F2[mid_width-n:mid_width+n+1,mid_height-n:mid_height+n+1] = 0 # select all but the first 50x50 (low) frequencies
    
    #inverse
    image1 = fp.ifft2(fp.ifftshift(F2)).real
    masked = cv2.bitwise_and(image1, image1, mask=frame)
    image1 = masked
    #cv2.imshow('After FFT',image1)
    cv2.imwrite('AfterFFT.png',image1)
    
    
    (thresh, im_bw) = cv2.threshold(frame, 110, 255, cv2.THRESH_BINARY)
    frame = im_bw
    return frame
  
def GetCorners(frame,color=255):
    white_pixels = np.array(np.where(frame == color))
    if color != 255:
        white_pixels = np.array(np.where(frame == color))
    arrx = white_pixels[0]
    arry = white_pixels[1]

    x1 = min(arrx)
    y1 = arry[(np.where(arrx==x1))[0][0]]
    x2 = max(arrx)
    y2 = arry[(np.where(arrx==x2))[0][0]]

    y3 = min(arry)
    x3 = arrx[(np.where(arry==y3))[0][0]]
    y4 = max(arry)
    x4 = arrx[(np.where(arry==y4))[0][0]]
    # if(color==255):
    #     x1+=10
    #     x4+=10
    #     y1-=10
    #     y4+=10

    #     x2-=10
    #     y2-=10
    #     x3-=10
    #     y3+=10

    # cv2.circle(original_image, (white_pixels[:,0][1], white_pixels[:,0][0]), 5, (0, 0, 255),-1)
    # cv2.circle(original_image, (white_pixels[:,-1][1], white_pixels[:,-1][0]), 5, (0, 255, 0,-1))

    corners = np.array([(y1,x1),(y3,x3),(y2,x2),(y4,x4)],np.int32)
    print('corners',corners)
    return corners

def RemoveBG(original_image,corners):
    rect = cv2.boundingRect(corners)
    x,y,w,h = rect
    white_bg = 255*np.ones_like(original_image)
    white_bg[y:y+h,x:x+w] = original_image[y:y+h,x:x+w]

    stencil  = np.zeros(original_image.shape[:-1]).astype(np.uint8)
    cv2.fillPoly(stencil, [corners], 255)
    sel = stencil != 255
    original_image[sel] = [255,255,255]
    return original_image





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
        original_image = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pos_frame = video_.get(cv2.CAP_PROP_POS_FRAMES)
        print(str(pos_frame)+" frames")
        frame = EdgeDetect(frame)
        corners = GetCorners(frame,255)
        original_image = RemoveBG(original_image,corners)
        kernel = np.ones((10,10), np.uint8)
        img_erosion = cv2.erode(original_image, kernel, iterations=1)
        original_image = cv2.dilate(original_image, kernel, iterations=1)
        cv2.imwrite('step1.png',original_image)
        frame = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        frame = EdgeDetect(frame)
        cornerAR = GetCorners(frame,0)
        plt.figure(figsize=(10,10))
        plt.imshow(frame)
        plt.show()
        cv2.polylines(original_image,[cornerAR],True,(0,255,0),3)
        pos_frame = video_.get(cv2.CAP_PROP_POS_FRAMES)
        print(str(pos_frame)+" frames")
        if pos_frame==1:
            cv2.imwrite("ARTagDetected.png",original_image)
    else:   # wait for next frame        
        video_.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
        cv2.waitKey(1000)
    if cv2.waitKey(10) == 27:
        break
    if video_.get(cv2.CAP_PROP_POS_FRAMES) == video_.get(cv2.CAP_PROP_FRAME_COUNT): 
        # traversed all frames
        break

 