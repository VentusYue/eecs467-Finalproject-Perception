import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
counter = 0
# stop_time = time.time()+10
while(True):
    # if stop_time >= time.time():
    #     break
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    counter += 1
    # Display the resulting frame
    # cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(counter)
        break

# When everything done, release the capture
print(counter)
cap.release()
cv2.destroyAllWindows()
