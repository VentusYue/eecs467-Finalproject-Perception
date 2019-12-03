from WebcamVideoStream import WebcamVideoStream
from FPS import FPS
import cv2

num_frames = 1000000

#
# print("[INFO] sampling frames from webcam...")
# stream = cv2.VideoCapture(0)
# fps = FPS().start()
#
# # loop over some frames
# while fps._numFrames < num_frames:
# 	# grab the frame from the stream and resize it to have a maximum
# 	# width of 400 pixels
# 	(grabbed, frame) = stream.read()
# 	# frame = imutils.resize(frame, width=400)
#
# 	# check to see if the frame should be displayed to our screen
# 	# if args["display"] > 0:
# 	# 	# cv2.imshow("Frame", frame)
# 	# 	key = cv2.waitKey(0) & 0xFF
#
# 	# update the FPS counter
# 	fps.update()
#
# # stop the timer and display FPS information
# fps.stop()
# print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#
# # do a bit of cleanup
# stream.release()
# cv2.destroyAllWindows()
#

# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()

# loop over some frames...this time using the threaded stream
while fps._numFrames < num_frames:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	# frame = imutils.resize(frame, width=400)

	# check to see if the frame should be displayed to our screen
	# if args["display"] > 0:
	# cv2.imshow("Frame", frame)

	# update the FPS counter
	fps.update()
	# if cv2.waitKey(0) & 0xFF == ord('q'):
	# 	break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
