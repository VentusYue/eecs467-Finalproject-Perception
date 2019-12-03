import cv2
import numpy as np
import time

def find_biggest_contour(image):
    image = image.copy()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # find largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask


filename = "image.jpg"
# filename = "incorrect1.jpg"

img = cv2.imread(filename)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# lower_orange = np.array([5, 0, 0])
# upper_orange = np.array([25, 200, 255])
# mask = cv2.inRange(hsv, lower_orange, upper_orange)

# lower_red = np.array([0, 50, 100])
# upper_red = np.array([10, 200, 255])

lower_red = np.array([0, 50, 100])
upper_red = np.array([8, 255, 200])
mask1 = cv2.inRange(hsv, lower_red, upper_red)

lower_red = np.array([170, 50, 100])
upper_red = np.array([180, 255, 200])
mask2 = cv2.inRange(hsv, lower_red, upper_red)
mask = cv2.bitwise_xor(mask1, mask2)
cv2.imwrite("./coloredmask.jpg", mask)


kernel = np.ones((3,3), np.uint8)
mask = cv2.erode(mask, kernel, iterations=2)
mask = cv2.dilate(mask, kernel, iterations=2)

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
if len(contours)==0:
    print("no ball detected")

for contour in contours:
    cv2.drawContours(img, contour, -1, (0, 255, 0), thickness = cv2.FILLED)

cv2.imwrite("./mask.jpg", mask)
cv2.imwrite("./contour.jpg", img)

_, mask = find_biggest_contour(mask)
cv2.imwrite("./largest_contour.jpg",mask)
gray = cv2.GaussianBlur(mask, (5, 5), 0)
cv2.imwrite("./blur.jpg",gray)
edges = cv2.Canny(gray, 50, 100)
cv2.imwrite("edges.jpg",edges)

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
cv2.imwrite("circles.jpg",img)
