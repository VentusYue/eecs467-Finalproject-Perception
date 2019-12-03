import argparse
import cv2
from FpsCounter import FpsCounter
from VideoGet import VideoGet
from VideoTracker import VideoTracker

def threadBoth(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for tracking object with ViedeoTracker object.
    Main thread serves only to pass frames between VideoGet and
    VideoTracker objects/threads.
    """

    video_getter = VideoGet(source).start()
    video_tracker = VideoTracker(video_getter.frame).start()
    # fps_main = FpsCounter().start()

    while True:
        if video_getter.stopped or video_tracker.stopped:
            video_tracker.stop()
            video_getter.stop()
            print("both stop")
            # print("main thread fps: {}".format(fps_main.counter()))
            break


        frame = video_getter.frame
        # video_tracker.frame = frame
        # fps_main.increment()

def main():
    threadBoth(0)

if __name__ == "__main__":
    main()
