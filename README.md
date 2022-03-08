# AR Tag Detection
<p align="center">
<img src="https://github.com/kavyadevd/ARTagDetection/blob/f9ce7cd4daae2ef46c01747de8a0295fe2912d8d/marker.png" alt="ARTAG" width="150px"></img></p>
<p>AprilTags are a type of fiducial marker. Fiducials, or more simply “markers,” are reference objects that are placed in the field of view of the camera when an image or video frame is captured. AprilTags are a specific type of fiducial marker, consisting of a black square with a white foreground that has been generated in a particular pattern</p>
<p>
Filters in image processing are just what the name suggests, Filter. They are typically a mask array of the same size as the original image which when superimposed on the ordinal image, extracts only the attributes that we are interested in.
</p>
<p></p>

## Part 1 – Detection
### AR Code detection:
Detecting the April Tag in any frame of [Tag1 video](https://drive.google.com/file/d/1EBSii403dwpq7xSvNH2HxUg48nqVUwQ6/view?usp=sharing) (just one frame).



## Terms:
1. Thresholding - Segmentation techniques in computer vision to separate the foreground from the background of the image. Basically selecting value T, all pixel intensities less than T = 0, and all pixel values greater than T = 255

References:
1. [AprilTag with Python](https://pyimagesearch.com/2020/11/02/apriltag-with-python/)
2. [FFT Edge Detection](https://wish-aks.medium.com/better-edge-detection-and-noise-reduction-in-images-using-fourier-transform-f85ed48b3123)
3. [Homography](https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html)
