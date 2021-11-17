from os import pardir
import sys
from PyQt5.QtCore import QPoint, Qt, QThread, QSize, QRect
from PyQt5.QtWidgets import QGroupBox, QMainWindow, QApplication, QPushButton, QCheckBox, QLabel, QFileDialog, QSpinBox, QComboBox, QCheckBox, QScrollArea, QWidget
from PyQt5 import uic
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage, QColor
import numpy as np
# import seam_carving
from seam_carving import run_seam_carving
import cv2

class MainBackgroundThread(QThread):
    def __init__(self):
        QThread.__init__(self)
        self.im = self.dx = self.dy = self.mode = self.vis = None
        self.mask = self.rmask = self.hremove = self.output = None

    def setting(self, im, dx, dy, mask, rmask, hremove, mode, vis):
        self.im = im
        self.dx = dx
        self.dy = dy
        self.mask = mask
        self.rmask = rmask
        self.hremove = hremove
        self.mode = mode
        self.vis = vis

    def getResult(self):
        return self.output

    def run(self):
        self.output = run_seam_carving(self.im, self.dx, self.dy,  mask=self.mask, rmask=self.rmask, hremove=self.hremove, mode=self.mode, vis=self.vis)
        if self.output is not None:
            self.output = self.output.astype(np.uint8)

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi('seam_carving_gui.ui', self)
        self.setWindowTitle("SeamCarving with PyQt5")
        #khởi tạo các object
        self.btOpen = self.findChild(QPushButton, 'bt_open')
        self.btSave = self.findChild(QPushButton, 'bt_save')
        self.btRun = self.findChild(QPushButton, 'bt_run')
        self.btClr = self.findChild(QPushButton, 'bt_clr')
        self.btUndo = self.findChild(QPushButton, 'bt_undo')
        self.ckbProtect = self.findChild(QCheckBox, 'ckb_protect')
        self.ckbRemove = self.findChild(QCheckBox, 'ckb_remove')
        self.ckbHremove = self.findChild(QCheckBox, 'ckb_hremove')
        self.wgImg = self.findChild(QWidget, 'wg_img')
        self.lbImg = self.findChild(QLabel, 'lb_img')
        self.lbMask = self.findChild(QLabel, 'lb_mask')
        self.sbCols = self.findChild(QSpinBox, 'sb_cols')
        self.sbRows = self.findChild(QSpinBox, 'sb_rows')
        self.sbSize = self.findChild(QSpinBox, 'sb_size')
        self.cmbMode = self.findChild(QComboBox, 'cmb_mode')
        self.ckbVis = self.findChild(QCheckBox, 'ckb_vis')
        self.lbInfor = self.findChild(QLabel, 'lb_infor')
        self.scrBar = self.findChild(QScrollArea, 'scr_area').verticalScrollBar()
        self.gbMask = self.findChild(QGroupBox, 'gb_mask')
        self.gbSeam = self.findChild(QGroupBox, 'gb_seam')
        self.worker = MainBackgroundThread()

        #cài đặt thuộc tính object và sự kiện
        self.btOpen.clicked.connect(self.OpenFile)
        self.btSave.clicked.connect(self.SaveFile)
        self.btRun.clicked.connect(self.RunSeam)
        self.btClr.clicked.connect(self.ClearMask)
        self.btUndo.clicked.connect(self.UndoMask)
        self.ckbProtect.clicked.connect(self.ToggleProtect)
        self.ckbRemove.clicked.connect(self.ToggleRemove)
        self.sbSize.valueChanged.connect(lambda x: self.setBrushSize(x))
        self.scrBar.rangeChanged.connect(lambda x,y: self.scrBar.setValue(y))
        self.worker.finished.connect(self.onFinished)
        self.lbMask.setGeometry(self.lbImg.rect())
        self.lbMask.hide()
        
        #khởi tạo các biến
        #lưu ảnh dưới dạng np.array
        self.img_in = None
        self.img_out = None

        #tạo qimage để vẽ ảnh cho lbmask
        self.Imask = QImage(self.lbImg.size(), QImage.Format_ARGB32)
        self.bgColor = QColor(0, 255, 255, 50)
        self.drawing = False
        self.brushSize = 10
        self.brushColor = Qt.black
        self.lastPoint = QPoint()

        self.show()

    def OpenFile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open File', '', "All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg)")
        if fname[0] != '':
            self.img_in = cv2.imdecode(np.fromfile(fname[0], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            self.showInfor('File input\n- Path: {}\n- Size: {}'.format(fname[0], self.img_in.shape))
            self.showImage(self.img_in)
            self.gbMask.setEnabled(True)
            self.gbSeam.setEnabled(True)
        else:
            self.showInfor('No file input !')

    def SaveFile(self):
        if self.img_out is None:
            if self.img_in is not None:
                self.img_out = self.img_in
            else:
                self.showInfor('No image to save.')
                return
        fname = QFileDialog.getSaveFileName(self, 'Save File', '', "JPG Files (*.jpg);;PNG Files (*.png)")
        if fname[0] != '':
            cv2.imencode(".jpg",self.img_out)[1].tofile(fname[0])
            self.showInfor(f'File save done !\n- Path: {fname[0]}')
        else:
            self.showInfor('File not save !')

    def RunSeam(self):
        if self.img_in is None:
            self.showInfor('No image to run.')
            return
        self.showInfor('START RUNNING !\nPLEASE WAIT !!!')
        dx = self.sbCols.value()
        dy = self.sbRows.value()
        mode = self.cmbMode.currentText()
        vis = self.ckbVis.isChecked()
        im = self.img_in
        mask = self.getMask(self.Imask, 255)
        rmask = self.getMask(self.Imask, 0)
        hremove = self.ckbHremove.isChecked()
        self.worker.setting(im, dx, dy, mask, rmask, hremove, mode, vis)
        self.worker.start()

    def ToggleProtect(self):
        if self.ckbProtect.isChecked():
            self.ckbRemove.setCheckState(0)
            self.lbMask.show()
            self.brushColor = Qt.white
        else:
            self.lbMask.hide()

    def ToggleRemove(self):
        if self.ckbRemove.isChecked():
            self.ckbProtect.setCheckState(0)
            self.lbMask.show()
            self.brushColor = Qt.black
        else:
            self.lbMask.hide()

    def onFinished(self):
        self.img_out = self.worker.getResult()
        if self.img_out is not None:
            self.showImage(self.img_out)
            self.showInfor('RUN DONE !\nFile output\n- Size: {}'.format(self.img_out.shape))
        else:
            self.showInfor('The image is not changed !')

    def showImage(self, img):
        pixmap = ArrayToQPixmap(img).scaled(self.wgImg.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbImg.setPixmap(pixmap)
        self.lbImg.adjustSize()
        self.resetMask()

    def resetMask(self):
        self.ckbProtect.setCheckState(0)
        self.ckbRemove.setCheckState(0) 
        self.lbMask.hide()
        self.lbMask.resize(self.lbImg.size())
        self.Imask = self.Imask.scaled(self.lbImg.size())
        self.Imask.fill(self.bgColor)
        self.lbMask.setPixmap(QPixmap.fromImage(self.Imask))

    def showInfor(self, new):
        old = self.lbInfor.text()
        self.lbInfor.setText(old+'\n---------------------------------------\n'+new)

    def mousePressEvent(self, event):
        # if left mouse button is pressed
        if event.button() == Qt.LeftButton:
            if not self.lbMask.isVisible():
                return
            # make drawing flag true
            self.drawing = True
            # make last point to the point of cursor
            self.lastPoint = event.pos()

            # creating painter object
            painter = QPainter(self.Imask)
            # set the pen of the painter
            painter.setPen(QPen(self.brushColor, self.brushSize,
                            Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPoint(event.pos())
            # update
            self.updateMask()

    def mouseMoveEvent(self, event):
        # checking if left button is pressed and drawing flag is true
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            # creating painter object
            painter = QPainter(self.Imask)
            # set the pen of the painter
            painter.setPen(QPen(self.brushColor, self.brushSize,
                            Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            # draw line from the last point of cursor to the current point
            # this will draw only one step
            painter.drawLine(self.lastPoint, event.pos())
            # print(self.lastPoint, self.drawing)
            # change the last point
            self.lastPoint = event.pos()
            # update
            self.updateMask()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            # make drawing flag false
            self.drawing = False

    def getMask(self, img, type):
        w = img.width()
        h = img.height()
        b = img.bits()
        b.setsize(w * h * 4)
        arr = np.frombuffer(b, dtype=np.uint8).reshape((h, w, 4))

        if type == 255:
            arr = arr[:,:,2:3]
        else:
            arr = arr[:,:,0:1]
        if (arr==0).all() or (arr==255).all():
            arr_gray = None 
        else: 
            arr_gray = arr.reshape(h, w)
        
        return arr_gray

    def ClearMask(self):
        self.Imask.fill(self.bgColor)
        self.updateMask()

    def UndoMask(self):
        # self.Imask.fill(self.bgColor)
        self.updateMask()
        return

    def updateMask(self):
        pixmap = QPixmap.fromImage(self.Imask)
        self.lbMask.setPixmap(pixmap)

    def setBrushSize(self, x):
        self.brushSize = x

def ArrayToQPixmap(cv_img):
    """Convert from an opencv image to QPixmap"""
    h, w, ch = cv_img.shape
    bytesPerLine = ch * w
    arr2 = np.require(cv_img, np.uint8, 'C')
    QImg = QImage(arr2, w, h, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
    pixmap = QPixmap.fromImage(QImg)
    return pixmap

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = UI()
    try:
        sys.exit(app.exec_())
    except:
        print('closing window...')