import cv2 as cv
import sys

img = cv.imread('apples.jpg')

if img is None:
    sys.exit("Could not read the image.")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
apples = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=200,
    param1=150, param2=20, minRadius=50, maxRadius=120)

for i in apples[0]:
    cv.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 2)  # draw the outer circle

cv.imshow('Apple detection', img)

cv.waitKey()
cv.destroyAllWindows()