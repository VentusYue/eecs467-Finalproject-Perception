#!/usr/bin/env python3
#
# python-v4l2capture
#
# This file is an example on how to capture a picture with
# python-v4l2capture.
#
# 2009, 2010 Fredrik Portstrom
#
# I, the copyright holder of this file, hereby release it into the
# public domain. This applies worldwide. In case this is not legally
# possible: I grant anyone the right to use this work for any
# purpose, without any conditions, unless such conditions are
# required by law.

from PIL import Image
import select
import v4l2capture
import cv2
import numpy as np
from collections import  deque
# Open the video device.
video = v4l2capture.Video_device("/dev/video0")

# Suggest an image size to the device. The device may choose and
# return another size if it doesn't support the suggested one.
size_x, size_y = video.set_format(1280, 1024)

# Create a buffer to store image data in. This must be done before
# calling 'start' if v4l2capture is compiled with libv4l2. Otherwise
# raises IOError.
video.create_buffers(1)

# Send the buffer to the device. Some devices require this to be done
# before calling 'start'.
video.queue_all_buffers()

# Start the device. This lights the LED if it's a camera that has one.
video.start()

# Wait for the device to fill the buffer.
select.select((video,), (), ())

image_data = video.read()
video.close()
image = Image.frombytes("RGB", (size_x, size_y), image_data)
image.save("image_original.jpg")

redLower = np.array([0, 0, 0])
redUpper = np.array([179, 255, 255])
mybuffer = 64
pts = deque(maxlen=mybuffer)
frame = cv2.imread("image_original.jpg")
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, redLower, redUpper)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
center = None
if len(cnts) > 0:
    c = max(cnts, key = cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
    if radius > 10:
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        pts.appendleft(center)
for i in range(1, len(pts)):
    if pts[i - 1] is None or pts[i] is None:
        continue
    thickness = int(np.sqrt(mybuffer / float(i + 1)) * 2.5)
    cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
cv2.imwrite('target.jpg', frame)
print("Saved target image.jpg (Size: " + str(size_x) + " x " + str(size_y) + ")")
