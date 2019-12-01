from threading import Thread
import cv2
import os
import lcm
os.sys.path.append('../lcmtypes/')
from lcmtypes import camera_pose_xyt_t

class VideoGet:
    """
    Class to get video stream from a usb camera using the mjpg format
    """

    def __init__(self, src=0):
        # initialize camera, set format to mjpeg
        self.stream = cv2.VideoCapture(src)
        self.stream.set(6, cv2.VideoWriter.fourcc('M','J','P','G'))
        self.stream.set(3, 640) # set resolution
        self.stream.set(4, 480)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
        self.stream.release()
