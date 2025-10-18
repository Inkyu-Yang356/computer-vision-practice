import cv2 as cv
import numpy as np
import sys

img = cv.imread('soccer.jpg')               # 영상 읽기
img_show = np.copy(img)         # 붓칠을 디스플레이할 목적의 영상

if img is None:
    print('Image load failed!')
    sys.exit()

mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
mask[:,:] = cv.GC_PR_BGD                    # 모든 화소를 배경일 것 같음으로 초기화

background=np.zeros((1,65), np.float64)
foreground=np.zeros((1,65), np.float64)

cv.grabCut(img, mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)
mask2 = np.where((mask==cv.GC_BGD) | (mask==cv.GC_PR_BGD), 1, 0).astype('uint8')
grab = img * mask2[:,:,np.newaxis]  # 마스크를 3채널로 변환하여 픽셀 곱셈
cv.imshow('Grab Cut image', grab)

cv.waitKey()
cv.destroyAllWindows()