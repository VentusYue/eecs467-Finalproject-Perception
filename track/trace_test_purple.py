import select
import v4l2capture
import time
import io
from PIL import Image
import cv2
import numpy as np

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

    lower_orange = np.array([50, 0, 50])
    upper_orange = np.array([255, 255, 200])
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
    if circles is not None:
        print("circles: ",len(circles[0]))
        max = 0

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
    return img

def main():
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
        image_data = video.read_and_queue()
        # image = Image.open(io.BytesIO(image_data))
        # img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        x = np.fromstring(image_data, dtype='uint8')
        #decode the array into an image
        img = cv2.imdecode(x, cv2.IMREAD_UNCHANGED)
        cv2.imwrite("frames/image_{}.jpg".format(counter),img)
        img_target = recognize_center(img)
        cv2.imwrite("targets/target_{}.jpg".format(counter),img_target)

        # k = cv2.waitKey(1) & 0xFF
        # # press 'q' to exit
        # if k == ord('q'):
        #     break
    print("Total frames: {}".format(counter))
    video.close()


if __name__ == '__main__':
    main()
