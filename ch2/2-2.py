import cv2 as cv
import sys

# Image Loading
img = cv.imread('soccer.jpg')

# Exception Check
if img is None:
    sys.exit('Not Found')

# Show Image
cv.imshow('Image Display', img)

# Wait for Key
cv.waitKey()
cv.destroyAllWindows()