import cv2 as cv
import sys

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('Image load failed!')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Convert to grayscale for edge detection
# The Sobel operator works on single channel images, so we use grayscale.

# Sobel Operator (Edge Detection)
# grad_x detects vertical edges (changes in intensity along x-axis)
grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)  # Sobel X, ksize = kernel size

# grad_y detects horizontal edges (changes in intensity along y-axis)
grad_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)  # Sobel Y, ksize = kernel size

# Convert to absolute values
sobel_x = cv.convertScaleAbs(grad_x)  # Scaler + Absolute
sobel_y = cv.convertScaleAbs(grad_y)

edge_strength = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)  # Combine both directions

cv.imshow('Original', gray)
cv.imshow('Sobel_X', sobel_x)
cv.imshow('Sobel_Y', sobel_y)
cv.imshow('Edge_Strength', edge_strength)

cv.waitKey()
cv.destroyAllWindows()