import cv2 as cv
import sys

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('Coult not read the image.')
    
cv.imshow('Display Image', img)
# cv.imshow('Upper left half', img[0:img.shape[0]//2, 0:img.shape[1]//2, :])

# XY 좌표상 25% ~ 75% 의 이미지를 표현 (단, 픽셀은 자연수)
cv.imshow('Upper left half', img[img.shape[0]//4:img.shape[0]*3//4, img.shape[1]//4:img.shape[1]*3//4, :])

# 각 색상 채널 표현 [2:R, 1:G, 0:B] (RGB 색상에 가까울수록 밝거나 어두움)
cv.imshow('Red Channel', img[0:img.shape[0]//2, 0:img.shape[1]//2, 2])
cv.imshow('Green Channel', img[0:img.shape[0]//2, 0:img.shape[1]//2, 1])
cv.imshow('Blue Channel', img[0:img.shape[0]//2, 0:img.shape[1]//2, 0])

cv.waitKey()
cv.destroyAllWindows()