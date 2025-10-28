import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import time
import sys

from sympy import sift

class SpecialEffect(QMainWindow):
    def __init__(self):
        super().__init__()
        self.imgA = cv.imread('mot_color70.jpg')
        self.imgB = cv.imread('mot_color83.jpg')
        self.grayA = cv.cvtColor(self.imgA, cv.COLOR_BGR2GRAY)
        self.grayB = cv.cvtColor(self.imgB, cv.COLOR_BGR2GRAY)
        
        self.setWindowTitle('SIFT matching')
        self.setGeometry(200, 200, 900, 200)  # 윈도우 너비 증가
        self.label = QLabel("SIFT matching test", self)
        self.label.setGeometry(10, 150, 880, 40)  # 레이블 너비와 높이 증가
        self.label.setWordWrap(True)  # 텍스트 줄바꿈 허용
        
        matchButton=QPushButton('matching', self)
        gaussianblurButton=QPushButton('gaussian blur', self)
        scaleinvarianceButton=QPushButton('scale invariance', self)
        quitButton=QPushButton('quit', self)

        matchButton.setGeometry(10, 10, 150, 30)
        gaussianblurButton.setGeometry(160, 10, 150, 30)
        scaleinvarianceButton.setGeometry(310, 10, 150, 30)
        quitButton.setGeometry(610, 10, 100, 30)  # Right align the quit button

        matchButton.clicked.connect(self.matchFunction)
        gaussianblurButton.clicked.connect(self.gaussianblurFunction)
        scaleinvarianceButton.clicked.connect(self.scaleinvarianceFunction)
        quitButton.clicked.connect(self.quitFunction)
    
    def matchFunction(self):
        """SIFT 특징점 매칭을 수행합니다."""
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.grayA, None)
        kp2, des2 = sift.detectAndCompute(self.grayB, None)

        if des1 is None or des2 is None:
            self.label.setText("특징점 없음")
            return

        start = time.time()
        flann_matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        knn_match = flann_matcher.knnMatch(des1, des2, 2)

        T = 0.6
        good_match = []
        for pair in knn_match:
            if len(pair) == 2:
                m, n = pair
                if (m.distance / n.distance) < T:
                    good_match.append(m)
        print('SIFT matching test')

        img_match = np.empty((max(self.imgA.shape[0], self.imgB.shape[0]),
                              self.imgA.shape[1] + self.imgB.shape[1], 3), dtype=np.uint8)
        cv.drawMatches(self.imgA, kp1, self.imgB, kp2, good_match, img_match,
                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # 왼쪽(이미지 A)에는 초록색 원, 오른쪽(이미지 B)에는 하늘색(RGB 135,206,235 -> BGR (235,206,135)) 원을 그림
        offset = self.imgA.shape[1]
        for m in good_match:
            pt1 = tuple(map(int, kp1[m.queryIdx].pt))
            pt2 = tuple(map(int, kp2[m.trainIdx].pt))
            cv.circle(img_match, pt1, 6, (0, 255, 0), 2, lineType=cv.LINE_AA)  # 초록색 원 (왼쪽)
            cv.circle(img_match, (pt2[0] + offset, pt2[1]), 6, (235, 206, 135), 2, lineType=cv.LINE_AA)  # 하늘색 원 (오른쪽)

        cv.imshow('Good Matches', img_match)

    def gaussianblurFunction(self):
        img_cut = self.imgA[190:350, 440:560, :]
        blurred = cv.GaussianBlur(img_cut, (0, 0), sigmaX=2)
        cv.imshow('blurred image', blurred)

    def scaleinvarianceFunction(self):
        """가우시안 블러를 적용한 후 SIFT 매칭 수 감소율을 계산합니다."""
        img_cut = self.imgB[190:350, 440:560, :]
        
        # 원본 이미지와 블러 이미지의 SIFT 매칭 수를 비교하여 감소율을 라벨에 표시하고 매칭 결과를 보여줍니다.
        sift = cv.SIFT_create()
        # 원본 매칭 계산
        kp1, des1 = sift.detectAndCompute(img_cut, None)
        kp2, des2 = sift.detectAndCompute(self.grayB, None)
        
        orig_count = 0
        T = 0.8
        if des1 is not None and des2 is not None:
            flann = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
            knn = flann.knnMatch(des1, des2, 2)
            good_orig = []
            for pair in knn:
                if len(pair) == 2:
                    m, n = pair
                    if (m.distance / n.distance) < T:
                        good_orig.append(m)
            orig_count = len(good_orig)
            # 블러 적용 및 매칭 계산
        blurred = cv.GaussianBlur(img_cut.copy(), (0, 0), sigmaX=2)
        gray_blur = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
        
        
        kp_b, des_b = sift.detectAndCompute(gray_blur, None)
        
        blur_count = 0
        good_blur = []
        if des_b is not None and des2 is not None:
            flann2 = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
            knn2 = flann2.knnMatch(des_b, des2, 2)
            for pair in knn2:
                if len(pair) == 2:
                    m, n = pair
                    if (m.distance / n.distance) < T:
                        good_blur.append(m)
            blur_count = len(good_blur)
            # 감소율 계산 및 라벨 업데이트
        if orig_count == 0:
            self.label.setText("원본 매칭 없음 — 감소율 계산 불가")
        else:
            decrease = (orig_count - blur_count) / orig_count * 100
            self.label.setText(f"블러 전 매칭 쌍: {orig_count}, 블러 후 매칭 쌍: {blur_count}, 매칭 쌍 감소율: {decrease:.1f}%")

            # 매칭 시각화
        img_match = np.empty((max(blurred.shape[0], self.imgB.shape[0]),
                              blurred.shape[1] + self.imgB.shape[1], 3), dtype=np.uint8)
        cv.drawMatches(blurred, kp_b if kp_b is not None else [], self.imgB, kp2 if kp2 is not None else [],
                       good_blur, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow('Blurred vs B Matches', img_match)

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()
        
app = QApplication(sys.argv)
win = SpecialEffect()
win.show()
app.exec_()