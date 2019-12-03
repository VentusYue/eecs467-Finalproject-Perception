import cv2
import imutils
import time

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

def takePicture():
    (grabbed, frame) = cap.read()
    showimg = frame
    cv2.imshow('img1', showimg)  # display the captured image
    cv2.waitKey(1)
    time.sleep(0.3) # Wait 300 miliseconds
    image = 'C:\Users\Lukas\Desktop\REMOVE\capture.png'
    cv2.imwrite(image, frame)
    cap.release()
    return image

print(takePicture())
