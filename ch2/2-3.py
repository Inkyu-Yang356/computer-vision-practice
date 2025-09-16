import cv2 as cv
import sys

# Image Loading
img = cv.imread('soccer.jpg')

# Exception Check
if img is None:
    sys.exit('Not Found')

# Convert Color and Size
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY) # RGB to Grayscale
gray_small = cv.resize(gray, dsize=(0,0), fx=0.5, fy=0.5)  # 50%로 축소

# Save Images
cv.imwrite('soccer_gray.png', gray)
cv.imwrite('soccer_gray_small.png', gray_small)

# Show Image
cv.imshow('Image Display', img)
cv.imshow('Gray Image', gray)
cv.imshow('Small Gray', gray_small)

# Wait for Key
cv.waitKey()
cv.destroyAllWindows()