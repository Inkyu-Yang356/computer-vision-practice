import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys

class SpecialEffect(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('effect')
        self.setGeometry(200, 200, 800, 200)
        self.label = QLabel("effect test", self)
        self.label.setGeometry(10, 150, 200, 30)
        
        readButton=QPushButton('read', self)
        viewButton=QPushButton('view', self)
        self.combo_box = QComboBox(self)
        self.combo_box.addItem("강하게")
        # 버튼 클릭 시 이 새 함수가 호출되도록 메서드 교체
        self.motionblurFunction = self.motionblur_with_strength
        self.combo_box.addItem("적절하게")
        self.combo_box.addItem("약하게")
        motionblurButton=QPushButton('motion blur', self)
        quitButton=QPushButton('quit', self)

        readButton.setGeometry(10, 10, 100, 30)
        viewButton.setGeometry(110, 10, 100, 30)
        self.combo_box.setGeometry(210, 10, 100, 30)
        motionblurButton.setGeometry(310, 10, 100, 30)
        quitButton.setGeometry(610, 10, 100, 30)  # Right align the quit button

        readButton.clicked.connect(self.pictureOpenFunction)
        viewButton.clicked.connect(self.viewFunction)
        motionblurButton.clicked.connect(self.motionblurFunction)
        quitButton.clicked.connect(self.quitFunction)
        
    def pictureOpenFunction(self):
        fname = QFileDialog.getOpenFileName(self, '사진 읽기', './img')
        self.img = cv.imread(fname[0])
        if self.img is None: sys.exit('파일을 읽을 수 없습니다.')
    
    def viewFunction(self):
        # View image from Read button
        cv.imshow('Original Image', self.img)
        
    def motionblurFunction(self):
        """길이 20의 수평 모션 블러 필터를 왼쪽 절반에만 적용"""
        img = self.img.copy()

        # 길이 20의 필터 생성
        kernel_size = 20
        kernel = np.zeros((1, kernel_size))
        kernel[:] = 1.0 / kernel_size  # [1/20, 1/20, ..., 1/20]

        # 블러 효과 적용
        blurred = cv.filter2D(img, -1, kernel)

        # 왼쪽 절반에만 적용
        h, w, _ = img.shape
        half = w // 2
        img[:, :half] = blurred[:, :half]

        # 결과 표시
        cv.imshow('Motion Blur Effect', img)
        
# 모션 블러 강도에 따라 동작하는 함수로 기존 메서드를 덮어쓰기
    def motionblur_with_strength(self):
        img = getattr(self, 'img', None)
        if img is None:
            return
        img_copy = img.copy()
        strength = self.combo_box.currentText()
        sizes = {'강하게': 40, '적절하게': 20, '약하게': 10}
        kernel_size = sizes.get(strength, 20)
        # 가로 방향의 모션 블러 커널 생성
        kernel = np.zeros((1, kernel_size), dtype=np.float32)
        kernel[:] = 1.0 / kernel_size
        blurred = cv.filter2D(img_copy, -1, kernel)
        h, w, _ = img_copy.shape
        half = w // 2
        img_copy[:, :half] = blurred[:, :half]
        # 결과 표시
        cv.imshow('Motion Blur Effect', img_copy)

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()
        
app = QApplication(sys.argv)
win = SpecialEffect()
win.show()
app.exec_()