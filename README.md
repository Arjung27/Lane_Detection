# Overview

This reository contains the code for detecting lanes in a image and in a video. We use the concept of classical computer vision to detect the lanes on the rectified image and then reproject the detections on the original image. After we reproject the detections we predct the turn.

# Dependencies
```
1. Python3.5
2. Ubuntu 16.04
3. OpenCV3.4.9
4. Numpy
5. Matplotlib
```

# Running the code

Please follow the following instructions to succesfully run the code:
```
1. Navigate to the parent directory where the code is present.

2. To run the code for image enhancement:
	python3 problem1.py <path_of_the_video>

3. To run the code for lane detection:
	python3 histogram_equalizer.py <path_to_the_file> <extension_of_the_file (mp4/png)>
```

The command (2) and (3) above generates the output video in the directory where the code is present.
