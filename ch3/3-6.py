import cv2 as cv
import sys

img = cv.imread('rose.png')

if img is None:
    sys.exit("Error: Image not found or path is incorrect.")

def interpolate(patch, method):
    patch_resized = cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation=method)
    cv.imshow(f'Resize {method}', patch_resized)

def makePatch(x,y,w,h):
    patch = img[y:y+h, x:x+w, :]
    cv.imshow('Patch', patch)
#    interpolate(patch, cv.INTER_NEAREST)
    interpolate(patch, cv.INTER_LINEAR)
#    interpolate(patch, cv.INTER_CUBIC)

cv.imshow('Original', img)
cv.setMouseCallback('Original', lambda event,x,y,flags,param: makePatch(x,y,200,200) if event==cv.EVENT_LBUTTONDOWN else None)

while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break