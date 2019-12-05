# Instructions on tuning detection

Wu Yue, 2019

## camera setting
```
python3 capture_picture.py
```

Check if the output image looks normal (no extreme exposure, weird color etc)
If you want to change the camera setting in linux, the following commands might be helpful

```
v4l2-ctl -d /dev/video0 --list-ctrls
v4l2-ctl --set-ctrl=exposure_auto=1
```

## Tuning HSV
```
python3 detect.py
```

[h,s,v] -> range[0:180, 0:255, 0:255]
1. s don't need to be changed
2. h, is most critical. 170:10 is red(5:20 is close to orange)
3. v, stands for lighting, 0:20, and 200:255 are likely ranges for extreme conditions,

##
```
python3 detect.py
```

[h,s,v] -> range[0:180, 0:255, 0:255]
1. s don't need to be changed
2. h, is most critical. 170:10 is red(5:20 is close to orange)
3. v, stands for lighting, 0:20, and 200:255 are likely ranges for extreme conditions,
