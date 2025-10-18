import cv2 as cv
import numpy as np

img = cv.imread('soccer.jpg')               # 영상 읽기
img_show = np.copy(img)         # 붓칠을 디스플레이할 목적의 영상

mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
mask[:,:] = cv.GC_PR_BGD                    # 모든 화소를 배경일 것 같음으로 초기화

BrushSiz = 9                                # 붓의 크기
LColor, RColor = (255,0,0), (0,0,255)       # 파란색(물체)과 빨간색(배경)

def painting(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img_show, (x,y), BrushSiz, LColor, -1)  # 왼쪽 버튼 클릭하면 파란색
        cv.circle(mask, (x,y), BrushSiz, cv.GC_FGD, -1)
    elif event == cv.EVENT_RBUTTONDOWN:
        cv.circle(img_show, (x,y), BrushSiz, RColor, -1)  # 오른쪽 버튼 클릭하면 빨간색
        cv.circle(mask, (x,y), BrushSiz, cv.GC_BGD, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img_show, (x,y), BrushSiz, LColor, -1)
                                                    # 왼쪽 버튼 클릭하고 이동하면 파란색
        cv.circle(mask, (x,y), BrushSiz, cv.GC_FGD, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
        cv.circle(img_show, (x,y), BrushSiz, RColor, -1)
                                                    # 오른쪽 버튼 클릭하고 이동하면 빨간색
        cv.circle(mask, (x,y), BrushSiz, cv.GC_BGD, -1)
    cv.imshow('Painting', img_show)

# img가 아닌 mask 위에 붓칠하는 코드

cv.namedWindow('Painting')
cv.setMouseCallback('Painting', painting)

while(True):
    if cv.waitKey(1) == ord('q'):           # 'q' 키 누르면 종료
        break

# 여기부터 GrabCut 적용하는 코드
background=np.zeros((1,65), np.float64)
foreground=np.zeros((1,65), np.float64)

# 마스크를 이용하여 GrabCut 수행 (grabCut의 initCount에 따라 경계선의 정확도가 달라짐)
cv.grabCut(img, mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)  # 5회 반복
mask2 = np.where((mask==cv.GC_BGD) | (mask==cv.GC_PR_BGD), 0, 1).astype('uint8')
grab = img * mask2[:,:,np.newaxis]
cv.imshow('Grab cut image', grab)

cv.waitKey()
cv.destroyAllWindows()