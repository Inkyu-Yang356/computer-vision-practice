import cv2 as cv
import numpy as np
import sys

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit("Error: Image not found or path is incorrect.")

img = cv.resize(img, dsize=(0,0), fx=0.25, fy=0.25)

def gamma(f, gamma=1.0):
    f1=f/255.0  # L=256이라고 가정
    return np.uint8((f1**gamma)*255)

gc = np.hstack([gamma(img, 0.5), gamma(img, 0.75), gamma(img, 1.0), gamma(img, 2.0),
                gamma(img, 3.0)])
    
cv.imshow('Gamma', gc)

cv.waitKey()
cv.destroyAllWindows()