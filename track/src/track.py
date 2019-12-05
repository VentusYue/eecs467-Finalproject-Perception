import numpy as np
import cv2
import platform
import sys
import time
import select
import io
from PIL import Image
import v4l2capture

TIME_SPAN = 10
SEARCH_SIZE = 40
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

from VideoGet import VideoGet
import lcm

import os
os.sys.path.append('../lcmtypes/')
from lcmtypes import camera_pose_xyt_t

def find_biggest_contour(image):
    image = image.copy()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # find largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask

def recognize_center(img, state_x,state_y):
    input_x = int(state_x.item())
    input_y = int(state_y.item())
    x_left = 0
    x_right = FRAME_WIDTH
    y_bottom = 0
    y_top = FRAME_HEIGHT

    if input_x - SEARCH_SIZE > 0:
        x_left = input_x - SEARCH_SIZE
    else:
        x_left = 0
    if input_x + SEARCH_SIZE < FRAME_WIDTH:
        x_right = input_x + SEARCH_SIZE
    else:
        x_right = FRAME_WIDTH
    if input_y - SEARCH_SIZE > 0:
        y_bottom = input_y - SEARCH_SIZE
    else:
        y_bottom = 0
    if input_y + SEARCH_SIZE < FRAME_HEIGHT:
        y_top = input_y + SEARCH_SIZE
    else:
        y_top = FRAME_HEIGHT


    # img_slice = img[x_left:x_right, y_bottom:y_top, :]
    img_slice = img[y_bottom:y_top, x_left:x_right, :]

    # print(img.shape)
    # print(input_x,input_y)
    # print(x_left,x_right,y_bottom,y_top)
    # cv2.imwrite("image.jpg",img)
    # cv2.imwrite("image_slice.jpg",img_slice)

    hsv = cv2.cvtColor(img_slice, cv2.COLOR_BGR2HSV)

    # lower_orange = np.array([2, 0, 50])
    # upper_orange = np.array([20, 255, 200])
    # mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # setting for the lab
    # lower_red = np.array([0, 50, 100])
    # upper_red = np.array([8, 255, 200])
    # mask1 = cv2.inRange(hsv, lower_red, upper_red)
    #
    # lower_red = np.array([170, 50, 100])
    # upper_red = np.array([180, 255, 200])
    # mask2 = cv2.inRange(hsv, lower_red, upper_red)
    # mask = cv2.bitwise_xor(mask1, mask2)

    # setting for the student lounge

    lower_red = np.array([0, 0, 20])
    upper_red = np.array([15, 255, 200])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 0, 20])
    upper_red = np.array([180, 255, 200])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.bitwise_xor(mask1, mask2)
    # cv2.imwrite("./coloredmask.jpg", mask)


    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))
    if len(contours)==0:
        print("no ball detected")
        return mask, img, (0,0,0)
    # for contour in contours:
    #     cv2.drawContours(img, contour, -1, (0, 255, 0), thickness = cv2.FILLED)
    # mask = max(contours, key=cv2.contourArea)
    # mask = contours
    # cv2.imwrite("./mask.jpg", mask)
    # cv2.imwrite("./contour.jpg", img)

    _, mask = find_biggest_contour(mask)
    # cv2.imwrite("./largest_contour.jpg",mask)
    gray = cv2.GaussianBlur(mask, (3,3), 0)
    # cv2.imwrite("./blur.jpg",gray)
    edges = cv2.Canny(mask1, 50, 100)
    # cv2.imwrite("edges.jpg",edges)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,
    1, minDist=100, param1=50, param2=15, minRadius=10, maxRadius=50)

    center_x = 0;
    center_y = 0;
    max = 0;
    if circles is not None:
        print("circles: ",len(circles[0]))

        for circle in circles[0]:
            x = int(circle[0])
            y = int(circle[1])
            r = int(circle[2])
            if r > max:
                max = r
                center_x = x
                center_y = y

            # cv2.circle(img, (x, y), r, (0, 0, 255), 3)
            # cv2.circle(img, (x, y), 3, (255, 255, 0), -1)
        center_x = x_left + center_x
        center_y = y_bottom + center_y
        # cv2.circle(img, (center_x, center_y), max, (0, 0, 255), 3)
        # cv2.circle(img, (center_x, center_y), 3, (255, 255, 0), -1)
    else:
        print("no circles")

    print("center of the target is: ({}, {}), radius: {}".format(center_x,center_y, max))
    # cv2.imwrite("circles.jpg",img)
    return mask, img, (center_x,center_y,max)

def recognize_center_without_EKF(img):
    # cv2.imwrite("test_img.jpg",img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower_orange = np.array([2, 0, 50])
    # upper_orange = np.array([20, 255, 200])
    # mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # setting for lab
    # lower_red = np.array([0, 50, 100])
    # upper_red = np.array([8, 255, 200])
    # mask1 = cv2.inRange(hsv, lower_red, upper_red)
    #
    # lower_red = np.array([170, 50, 100])
    # upper_red = np.array([180, 255, 200])
    # mask2 = cv2.inRange(hsv, lower_red, upper_red)
    # mask = cv2.bitwise_xor(mask1, mask2)

    # setting for student lounge
    lower_red = np.array([0, 0, 20])
    upper_red = np.array([15, 255, 200])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 0, 20])
    upper_red = np.array([180, 255, 200])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.bitwise_xor(mask1, mask2)
    # cv2.imwrite("./coloredmask.jpg", mask)
    #
    #
    #
    # cv2.imwrite("coloredmask.jpg",mask)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))
    if len(contours)==0:
        print("no ball detected")
        return mask, img, (0,0,0)
    for contour in contours:
        cv2.drawContours(img, contour, -1, (0, 255, 0), thickness = cv2.FILLED)

    # cv2.imwrite("./mask.jpg", mask)
    # cv2.imwrite("./contour.jpg", img)

    _, mask = find_biggest_contour(mask)
    # cv2.imwrite("./largest_contour.jpg",mask)
    gray = cv2.GaussianBlur(mask, (3,3), 0)
    # cv2.imwrite("./blur.jpg",gray)
    edges = cv2.Canny(gray, 50, 100)
    # cv2.imwrite("edges.jpg",edges)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,
        1, minDist=100, param1=50, param2=15, minRadius=10, maxRadius=50)

    center_x = 0;
    center_y = 0;
    max = 0;
    if circles is not None:
        print("circles: ",len(circles[0]))

        for circle in circles[0]:
            x = int(circle[0])
            y = int(circle[1])
            r = int(circle[2])
            if r > max:
                max = r
                center_x = x
                center_y = y

            # cv2.circle(img, (x, y), r, (0, 0, 255), 3)
            # cv2.circle(img, (x, y), 3, (255, 255, 0), -1)
        # cv2.circle(img, (center_x, center_y), max, (0, 0, 255), 3)
        # cv2.circle(img, (center_x, center_y), 3, (255, 255, 0), -1)
    else:
        print("no circles")
        # cv2.imwrite("../Detection/no-detection.jpg",img)

    print("center of the target is: ({}, {}), radius: {}".format(center_x,center_y, max))
    # cv2.imwrite("circles.jpg",img)
    return mask, img, (center_x,center_y,max)


def run_EKF_model(state, P, Q, dt):

    # model is
    # X(0) = X(0) + X(2)*dt
    # X(1) = X(1) + X(3)*dt
    # X(2) = X(2)
    # X(3) = X(3)
    # it has no input, so Ju = 0

    state[0] = state[0] + dt*state[2]
    state[1] = state[1] + dt*state[3]
    state[2] = state[2]
    state[3] = state[3]

    # covariance matrix gets updated through
    # P = J*P*trans(J) + Q
    # where J = [1, 0, dt, 0;
    #			 0, 1, 0, dt;
    #			 0, 0, 1, 0;
    #			 0, 0, 0, 1]

    J = np.matrix('1.0,0.0,0.0,0.0;\
                   0.0,1.0,0.0,0.0;\
                   0.0,0.0,1.0,0.0;\
                   0.0,0.0,0.0,1.0')
    J[0,2] = dt
    J[1,3] = dt
    P = J*P*(J.transpose()) + Q

    return state, P, J

def run_EKF_measurement(state,measurement, P):
    # Observation is (x,y) = (X(0), X(1))
    # H = [1, 0, 0, 0, 0, 0;
    # 		0, 1, 0, 0, 0, 0]
    # R = [sigma_x, 0;
    #		0, sigma_y]

    H = np.matrix('1.0,0.0,0.0,0.0; \
                   0.0,1.0,0.0,0.0')


    R = np.matrix('5.0,0.0;\
                    0.0,5.0')

    z = measurement - H*state
    HPH = H*P*(H.transpose())
    S = HPH + R
    invS = np.linalg.inv(S)
    K = P*(H.transpose())*np.linalg.inv(S)
    state = state + K*z
    P = P - P*(H.transpose())*np.linalg.inv(S)*H*P
    debug_print = False

    if (debug_print == True):
        print ('running new measurement')
        print ("norm P is: {}".format(np.linalg.norm(P)))
        print ("z is {}".format(z.transpose()))
        print ("HPH is {}".format(HPH))
        print ("S is {}".format(S))
        print ("invS is {}".format(invS))
        print ("PH is {}".format(P*(H.transpose())))
        print ("K is {}".format(K))

    return state, P


def transform_camera_to_2d(pixel_coord):
    # RMS Error: 0.1744614622584057
    # camera matrix:
    #  [[610.20180181   0.         338.32138155]
    #  [  0.         608.82221759 233.43050486]
    #  [  0.           0.           1.        ]]
    # distortion coefficients:
    #  [-4.48105576e-01  2.83494222e-01  1.45578074e-04 -2.10300503e-04
    #  -1.35291600e-01]

    fx = 610
    cx = 320
    cy = 240
    intrinsic = np.array([[fx,0.0,cx,0.0],
                [0.0,fx, cy,0.0],
                [0.0,0.0, 1.0,0.0]] )
    intrinsic_inv = np.linalg.inv(intrinsic)
    print("s")
    return intrinsic_inv @ pixel_coord


def main():
    # lcm
    lc = lcm.LCM()

    # Global Variables
    state = np.matrix('0.0;0.0;0.0;0.0') # x, y, xd, yd,

    # P and Q matrices for EKF
    P = np.matrix('10.0,0.0,0.0,0.0; \
                0.0,10.0,0.0,0.0; \
                0.0,0.0,10.0,0.0; \
                0.0,0.0,0.0,10.0' )


    Q = np.matrix('2.0,0.0,0.0,0.0; \
                0.0,2.0,0.0,0.0; \
                0.0,0.0,2.0,0.0; \
                0.0,0.0,0.0,2.0')

    measurement = np.matrix('0;0')
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # print basic info
    print('python ' + platform.python_version())
    print('opencv ' + cv2.__version__)
    print('numpy ' + np.version.version)

    # lauch getter thread
    print("Start sampling THREADED frames from webcam...")
    vs = VideoGet(src=0).start()

    stop_time = time.time() + TIME_SPAN
    start_time = time.time()

    prev_time = time.time()
    i = 0

    while(True):
        if time.time() > stop_time:
            break

        now_time = time.time()
        dt = now_time - prev_time

        i+=1
        # run the model every 0.01 s
        if (dt > 0.001):
            prev_time = now_time

            state, P, J = run_EKF_model(state, P, Q, dt)

        # read camera
        # ret, frame = cap.read()
        frame = vs.read()
        ret = True;
        print("frame: {}".format(i))
        if ret == True:

            # For initilization, process the whole image, otherwise, utilize the predicted position
            if i < 10:
                mask, cimg, (x,y,r) = recognize_center_without_EKF(frame)
            else:
                mask, cimg, (x,y,r) = recognize_center(frame,state[0],state[1])

            # if i == 5:
            #     break
            # if x==0:
            #     continue
            measurement[0] = x
            measurement[1] = y
            if(measurement[0] != 0) and (measurement[1] != 0):
                print("run EKF")
                state, P = run_EKF_measurement(state, measurement, P)
            else:
                print("no motion detected, continue")

                # continue
            print("x: {}, y:{}. state 0: {}, state 1: {}".format(x,y,state[0],state[1]))
            # if(x != 0):
            #     cv2.circle(cimg, (int(x), int(y)), 50, (255), 5)
            #
            # if(state[0] != 0):
            #     cv2.circle(cimg, (int(state[0]),int(state[1])), 20, (255), 3)

            # pixel_coord = np.array([state[0],state[1],1])
            # world_2d_coord = transform_camera_to_2d(pixel_coord)
            # print(world_2d_coord)
            # cv2.imshow('all',cimg)
            msg = camera_pose_xyt_t()
            msg.x = state[0]
            msg.y = state[1]
            lc.publish("CAMERA_POSE_CHANNEL", msg.encode())

        # close
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break


    print("Time {}, frames: {}".format(time.time()-start_time, i))
    # clean up
    # vs.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
