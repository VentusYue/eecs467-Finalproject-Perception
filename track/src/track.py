import argparse
import cv2
from CountsPerSec import CountsPerSec
from VideoGet import VideoGet
from VideoTracker import VideoTracker

def threadBoth(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """

    video_getter = VideoGet(source).start()
    video_tracker = VideoTracker(video_getter.frame).start()
    cps = CountsPerSec().start()

    while True:
        if video_getter.stopped or video_tracker.stopped:
            video_tracker.stop()
            video_getter.stop()
            break
        # print(cps.counter())

        frame = video_getter.frame
        frame = putIterationsPerSec(frame, cps.countsPerSec())
        video_tracker.frame = frame
        cps.increment()

def main():
    threadBoth(0)

if __name__ == "__main__":
    main()
