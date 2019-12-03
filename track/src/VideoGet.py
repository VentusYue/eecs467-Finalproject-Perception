from threading import Thread
import cv2
import os
import time
from FpsCounter import FpsCounter

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
        self.i = 0
        # self.fps_getter = FpsCounter()

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        # self.fps_getter.start()
        while not self.stopped:
            self.i += 1
            # if self.fps_getter.end():
            #     self.stop()
            if not self.grabbed:
                self.stop()
            else:
                # self.fps_getter.increment()

                # print("Getter update",i)
                (self.grabbed, self.frame) = self.stream.read()
                # print(self.grabbed)
    def read(self):
        return self.frame

    def stop(self):
        print("Frames: {}".format(self.i))
        self.stopped = True
        # print("Getter fps: {}".format(self.fps_getter.fps()))
        self.stream.release()

# video_getter = VideoGet(0).start()
# video_getter.get()
