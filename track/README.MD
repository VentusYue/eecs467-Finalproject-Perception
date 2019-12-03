# Instructions

Wu Yue, 2019

The code hasn't been reorganized now.

## Detection Tuning
Put the colored ball in camera's sight, and do the following:
```
cd ./Detection
python3 capture_picture.py
```
Then launch the detection code, which would generate results with each processing step
```
python3 detect.py
```
Notice that hsv range is the most critical for detection
## Tracking Tuning
```
python3 src/track.py
```
There are two critical global parameters to be noticed
1. SEARCH_SIZE:
area of pixels to search for the kalman filter,
decrease the size would: 1. speed up fps, 2. narrow the horizon.
40 is a sample value for our pendulum
2. TIME_SPAN:
run time for tracking program (in second),
I haven't figure out how to kill thread properly, please ctrl+c to stop it for now
