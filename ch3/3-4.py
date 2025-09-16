import cv2 as cv
import matplotlib.pyplot as plt
import sys

img = cv.imread('mistyroad.jpg')

if img is None:
    sys.exit("Error: Image not found or path is incorrect.")
    
# Convert to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

# Histogram Caclulation
h = cv.calcHist([gray], channels=[0], mask=None, histSize=[256], ranges=[0,256])
plt.plot(h, color='r', linewidth=1), plt.show()

# Histogram Equalization
equal = cv.equalizeHist(gray)
plt.imshow(equal, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

# Equalized Histogram Calculation
h2 = cv.calcHist([equal], channels=[0], mask=None, histSize=[256], ranges=[0,256])
plt.plot(h2, color='r', linewidth=1), plt.show()

cv.waitKey()
cv.destroyAllWindows()