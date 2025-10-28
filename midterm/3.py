import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import time
import sys

from sympy import sift

class SpecialEffect(QMainWindow):
    def __init__(self):
        super().__init__()
        self.imgA = cv.imread('elder.png')
        self.imgB = cv.imread('traffic1.jpg')
        self.grayA = cv.cvtColor(self.imgA, cv.COLOR_BGR2GRAY)
        self.grayB = cv.cvtColor(self.imgB, cv.COLOR_BGR2GRAY)
        self.current_img = self.imgB  # 현재 선택된 이미지를 추적
        
        self.setWindowTitle('homograpy agent')
        self.setGeometry(200, 200, 900, 200)  # 윈도우 너비 증가
        self.label = QLabel("SIFT matching test", self)
        self.label.setGeometry(10, 150, 880, 40)  # 레이블 너비와 높이 증가
        self.label.setWordWrap(True)  # 텍스트 줄바꿈 허용
        
        viewsignButton=QPushButton('view sign', self)
        viewroadButton=QPushButton('view road', self)
        edgeButton=QPushButton('edge', self)
        homographyButton=QPushButton('homography', self)
        quitButton=QPushButton('quit', self)

        viewsignButton.setGeometry(10, 10, 150, 30)
        viewroadButton.setGeometry(160, 10, 150, 30)
        edgeButton.setGeometry(310, 10, 150, 30)
        homographyButton.setGeometry(460, 10, 150, 30)
        quitButton.setGeometry(610, 10, 100, 30)  # Right align the quit button

        viewsignButton.clicked.connect(self.viewSignFunction)
        viewroadButton.clicked.connect(self.viewRoadFunction)
        edgeButton.clicked.connect(self.edgeFunction)
        homographyButton.clicked.connect(self.homographyFunction)
        quitButton.clicked.connect(self.quitFunction)
    
    def viewSignFunction(self):
        """교통 표지판 영상 표시"""
        self.current_img = self.imgA  # elder.png (교통표지판)
        cv.imshow('Sign Image', self.current_img)
    
    def viewRoadFunction(self):
        """도로 영상 표시"""
        self.current_img = self.imgB  # traffic1.jpg (도로)
        cv.imshow('Road Image', self.current_img)
    
    def edgeFunction(self):
        """현재 선택된 이미지에 Canny Edge 적용"""
        if hasattr(self, 'current_img') and self.current_img is not None:
            # 현재 이미지를 그레이스케일로 변환
            if len(self.current_img.shape) == 3:
                gray = cv.cvtColor(self.current_img, cv.COLOR_BGR2GRAY)
            else:
                gray = self.current_img
            
            # Canny Edge 적용
            edges = cv.Canny(gray, 100, 200)
            
            # 결과 표시
            cv.imshow('Canny Edge', edges)
        else:
            self.label.setText("먼저 이미지를 선택하세요 (view sign 또는 view road)")
    
    def homographyFunction(self):
        """homography 기능 - 엣지 픽셀이 5% 이상인 경우 RANSAC 호모그래피 추정"""
        if not hasattr(self, 'current_img') or self.current_img is None:
            self.label.setText("먼저 이미지를 선택하세요 (view sign 또는 view road)")
            return
        
        # 현재 이미지를 그레이스케일로 변환
        if len(self.current_img.shape) == 3:
            gray = cv.cvtColor(self.current_img, cv.COLOR_BGR2GRAY)
        else:
            gray = self.current_img
        
        # Canny Edge 적용
        edges = cv.Canny(gray, 100, 200)
        
        # 전체 픽셀 수와 엣지 픽셀 수 계산
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_pixels = np.count_nonzero(edges)
        edge_percentage = (edge_pixels / total_pixels) * 100
        
        # 엣지 픽셀이 5% 미만인 경우
        if edge_percentage < 5.0:
            self.label.setText("호모그래피 추정 실패 가능성이 높음")
            return
        
        # 엣지 픽셀이 5% 이상인 경우 호모그래피 추정
        try:
            # SIFT 특징점 검출
            sift = cv.SIFT_create()
            kp1, des1 = sift.detectAndCompute(self.grayA, None)  # elder.png (참조 이미지)
            kp2, des2 = sift.detectAndCompute(gray, None)        # 현재 선택된 이미지
            
            if des1 is None or des2 is None:
                self.label.setText("특징점 검출 실패 - 호모그래피 추정 불가")
                return
            
            # FLANN 매칭
            flann = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
            knn_matches = flann.knnMatch(des1, des2, 2)
            
            # 좋은 매칭점 필터링 (Lowe's ratio test)
            good_matches = []
            for match_pair in knn_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 4:
                self.label.setText(f"매칭점 부족 ({len(good_matches)}개) - 호모그래피 추정 불가")
                return
            
            # 매칭점에서 좌표 추출
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # RANSAC을 사용한 호모그래피 행렬 추정
            homography_matrix, mask = cv.findHomography(src_pts, dst_pts, 
                                                       cv.RANSAC, 
                                                       ransacReprojThreshold=5.0)
            
            if homography_matrix is None:
                self.label.setText("호모그래피 행렬 추정 실패")
                return
            
            # 호모그래피를 사용하여 이미지 정렬
            h, w = self.current_img.shape[:2]
            aligned_img = cv.warpPerspective(self.imgA, homography_matrix, (w, h))
            
            # 정렬된 이미지의 R채널 추출
            if len(aligned_img.shape) == 3:
                r_channel = aligned_img[:, :, 2]  # OpenCV는 BGR 순서이므로 R채널은 인덱스 2
            else:
                r_channel = aligned_img  # 이미 그레이스케일인 경우
            
            # 결과 표시 (R채널을 grayscale로)
            cv.imshow('Aligned R Channel (Grayscale)', r_channel)
            
            # 인라이어 개수 계산
            inliers = np.sum(mask)
            self.label.setText(f"호모그래피 추정 성공 (엣지: {edge_percentage:.2f}%, 매칭점: {len(good_matches)}개, 인라이어: {inliers}개)")
            
        except Exception as e:
            self.label.setText(f"호모그래피 추정 오류: {str(e)}")
    
    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()
        
app = QApplication(sys.argv)
win = SpecialEffect()
win.show()
app.exec_()