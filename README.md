# AR Tag Detection
<p align="center">
<img src="https://github.com/kavyadevd/ARTagDetection/blob/f9ce7cd4daae2ef46c01747de8a0295fe2912d8d/marker.png" alt="ARTAG" width="150px"></img></p>
<p>AprilTags are a type of fiducial marker. Fiducials, or more simply “markers,” are reference objects that are placed in the field of view of the camera when an image or video frame is captured. AprilTags are a specific type of fiducial marker, consisting of a black square with a white foreground that has been generated in a particular pattern</p>
<p>
Filters in image processing are just what the name suggests, Filter. They are typically a mask array of the same size as the original image which when superimposed on the ordinal image, extracts only the attributes that we are interested in.
</p>
<p></p>


## Executing the code

### File structure

    .
    ├── ARdetect.py                             # python code file for problem 1.a and 1.b
    ├── TestudoSuperimpose.py                   # python code file for problem 2.a
    ├── CubeAR.py                               # python code file for problem 2.b
    ├── Results                                 # contains output images
    ├── 1tagvideo.mp4                           # input video file
    ├── testudo.png                             # image to be superimposed
    └── README.md


### Commands:
1. For executing AR TAG detect and decode ( Problem 1.a and 1.b.)

```
python3 Ardetect.py
```

2. For executing image superimposition ( Problem 2.a )

```
python3 TestudoSuperimpose.py
```

3. For executing solution to AR TAG detect and decode ( Problem 2.b )

```
python3 CubeAR.py
```
To close the output window at any time press esc


## Part 1 – Detection
### AR Code detection:
Detecting the April Tag in any frame of [Tag1 video](https://drive.google.com/file/d/1EBSii403dwpq7xSvNH2HxUg48nqVUwQ6/view?usp=sharing) (just one frame).

## Part 2 – Tracking
Track the detected AR Tag and superimpose an image and then a 3D cube while maintaining the orientation

## Terms:
1. Thresholding - Segmentation techniques in computer vision to separate the foreground from the background of the image. Basically selecting value T, all pixel intensities less than T = 0, and all pixel values greater than T = 255

References:
1. [AprilTag with Python](https://pyimagesearch.com/2020/11/02/apriltag-with-python/)
2. [FFT Edge Detection](https://wish-aks.medium.com/better-edge-detection-and-noise-reduction-in-images-using-fourier-transform-f85ed48b3123)
3. [Homography](https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html)
