import cv2 as cv
import sys

# Image Loading
img = cv.imread('girl_laughing.jpg')

# Exception Check
if img is None:
    sys.exit('Not Found')

# Drawing
cv.rectangle(img, (830, 30), (1000,200), (0,0,255), 2)
cv.putText(img, 'laugh', (830, 24), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

# Show Image
cv.imshow('Draw', img)

# Wait for Key
cv.waitKey()
cv.destroyAllWindows()