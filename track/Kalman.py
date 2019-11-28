import numpy as np
import cv2
import platform
import sys
import time
import select
import io
from PIL import Image
import v4l2capture




def find_biggest_contour(image):
    image = image.copy()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # find largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask

def recognize_center(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([2, 0, 50])
    upper_orange = np.array([20, 255, 200])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))
    for contour in contours:
        cv2.drawContours(img, contour, -1, (0, 255, 0), thickness = cv2.FILLED)

    # cv2.imwrite("./mask.jpg", mask)
    # cv2.imwrite("./contour.jpg", img)

    _, mask = find_biggest_contour(mask)
    # cv2.imwrite("./largest_contour.jpg",mask)
    gray = cv2.GaussianBlur(mask, (5,5), 0)
    # cv2.imwrite("./blur.jpg",gray)
    edges = cv2.Canny(gray, 50, 100)
    # cv2.imwrite("edges.jpg",edges)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,
        1, 100, param1=50, param2=20, minRadius=10, maxRadius=500)

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
        cv2.circle(img, (center_x, center_y), max, (0, 0, 255), 3)
        cv2.circle(img, (center_x, center_y), 3, (255, 255, 0), -1)
    else:
        print("no circles")

    print("center of the target is: ({}, {})".format(center_x,center_y))
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



    # def main():
    # open camera

    video = v4l2capture.Video_device("/dev/video0")
    size_x, size_y = video.set_format(640, 480, fourcc='MJPG')
    video.create_buffers(60)
    video.queue_all_buffers()
    video.start()
    # stop_time = time.time() + 500

    prev_time = time.time()
    i = 0

    while(True):

        now_time = time.time()
        dt = now_time - prev_time

        i+=1
        # run the model every 0.01 s
        if (dt > 0.01):
            prev_time = now_time

            state, P, J = run_EKF_model(state, P, Q, dt)

        # read camera
        # ret, frame = cap.read()
        select.select((video,), (), ())
        image_data = video.read_and_queue()
        raw_image = np.fromstring(image_data, dtype='uint8')
        frame = cv2.imdecode(raw_image, cv2.IMREAD_UNCHANGED)
        ret = True;
        print("frame: {}".format(i))
        if ret == True:
            # process
            mask, cimg, (x,y,r) = recognize_center(frame)
            # if x==0:
            #     continue
            measurement[0] = x
            measurement[1] = y
            if(measurement[0] != 0) and (measurement[1] != 0):
                print("run EKF")
                state, P = run_EKF_measurement(state, measurement, P)
            print("x: {}, state 0: {}".format(x,state[0]))
            if(x != 0):
                cv2.circle(cimg, (int(x), int(y)), 50, (255), 5)

            if(state[0] != 0):
                cv2.circle(cimg, (int(state[0]),int(state[1])), 20, (255), 3)

            pixel_coord = np.array([state[0],state[1],1])
            world_2d_coord = transform_camera_to_2d(pixel_coord)
            print(world_2d_coord)
            cv2.imshow('all',cimg)

        # close
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # clean up
    video.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
