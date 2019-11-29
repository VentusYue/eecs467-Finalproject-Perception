#!/usr/bin/env python3
#
# python-v4l2capture
#
# This file is an example on how to capture a mjpeg video with
# python-v4l2capture.
#
# 2009, 2010 Fredrik Portstrom
#
# I, the copyright holder of this file, hereby release it into the
# public domain. This applies worldwide. In case this is not legally
# possible: I grant anyone the right to use this work for any
# purpose, without any conditions, unless such conditions are
# required by law.


import select
import v4l2capture
import time

# Open the video device.
video = v4l2capture.Video_device("/dev/video0")

size_x, size_y = video.set_format(640, 480, fourcc='MJPG')

video.create_buffers(60)

video.queue_all_buffers()

video.start()

stop_time = time.time() + 5
counter = 0
while stop_time >= time.time():
    # Wait for the device to fill the buffer.
    select.select((video,), (), ())
    counter += 1
    # The rest is easy :-)
    image_data = video.read_and_queue()

print(counter)
video.close()
# print("Saved video.mjpg (Size: " + str(size_x) + " x " + str(size_y) + ")")
