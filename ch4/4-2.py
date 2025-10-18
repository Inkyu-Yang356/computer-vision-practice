import cv2 as cv
import sys

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit("Could not read the image.")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply Canny edge detection with two different threshold sets. (Example: Car Plate Recognition)
canny1 = cv.Canny(gray, 50, 150)  # Lower thresholds: detects more edges, may include noise
canny2 = cv.Canny(gray, 150, 250) # Higher thresholds: detects fewer, more prominent edges

cv.imshow('Original', gray)
cv.imshow('Canny1', canny1)
cv.imshow('Canny2', canny2)

cv.waitKey()
cv.destroyAllWindows()