import sys
import webbrowser
from PyQt5.QtCore import QPoint, Qt, QThread, QUrl
from PyQt5.QtWidgets import QAction, QGroupBox, QMainWindow, QApplication, QPushButton, QCheckBox, QLabel, QFileDialog, QSpinBox, QComboBox, QCheckBox, QScrollArea, QWidget, QMessageBox
from PyQt5 import uic
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage, QColor
import numpy as np
from seam_carving import run_seam_carving
import cv2
import time

class MainBackgroundThread(QThread):
    def __init__(self):
        QThread.__init__(self)
        self.im = self.dx = self.dy = self.mode = self.vis = self.vismask = self.visall = None
        self.vismap = self.mask = self.rmask = self.hremove = self.kratio = self.output = self.all_seam = None

    def setting(self, im, dx, dy, mask, rmask, hremove, kratio, mode, vis, vismask, vismap, visall, downsize):
        self.im = im
        self.dx = dx
        self.dy = dy
        self.mask = mask
        self.rmask = rmask
        self.hremove = hremove
        self.kratio = kratio
        self.mode = mode
        self.vis = vis
        self.vismask = vismask
        self.vismap = vismap
        self.visall = visall
        self.downsize = downsize

    def getResult(self):
        return self.output, self.all_seam

    def run(self):
        self.output, self.all_seam = run_seam_carving(self.im, self.dx, self.dy,  mask=self.mask, rmask=self.rmask, hremove=self.hremove, kratio=self.kratio, mode=self.mode, vis=self.vis, vismask=self.vismask, vismap=self.vismap, visall=self.visall, downsize=self.downsize)
        if self.output is not None:
            self.output = self.output.astype(np.uint8)

    def stop(self):
        self.terminate()
        cv2.destroyAllWindows()

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi('seam_carving_gui.ui', self)
        self.setWindowTitle("SeamCarving with PyQt5")

        #initialize objects
        self.btOpen = self.findChild(QPushButton, 'bt_open')
        self.btSave = self.findChild(QPushButton, 'bt_save')
        self.btRun = self.findChild(QPushButton, 'bt_run')
        self.btClr = self.findChild(QPushButton, 'bt_clr')
        self.btUndo = self.findChild(QPushButton, 'bt_undo')
        self.ckbProtect = self.findChild(QCheckBox, 'ckb_protect')
        self.ckbRemove = self.findChild(QCheckBox, 'ckb_remove')
        self.ckbHremove = self.findChild(QCheckBox, 'ckb_hremove')
        self.ckbRatio = self.findChild(QCheckBox, 'ckb_ratio')
        self.wgImg = self.findChild(QWidget, 'wg_img')
        self.lbImg = self.findChild(QLabel, 'lb_img')
        self.lbMask = self.findChild(QLabel, 'lb_mask')
        self.sbCols = self.findChild(QSpinBox, 'sb_cols')
        self.sbRows = self.findChild(QSpinBox, 'sb_rows')
        self.sbSize = self.findChild(QSpinBox, 'sb_size')
        self.ckbDownsize = self.findChild(QCheckBox, 'ckb_downsize')
        self.cmbMode = self.findChild(QComboBox, 'cmb_mode')
        self.ckbProcess = self.findChild(QCheckBox, 'ckb_process')
        self.ckbMask = self.findChild(QCheckBox, 'ckb_mask')
        self.ckbMap = self.findChild(QCheckBox, 'ckb_energy_map')
        self.ckbASeam = self.findChild(QCheckBox, 'ckb_all_seam')
        self.lbInfor = self.findChild(QLabel, 'lb_infor')
        self.scrBar = self.findChild(QScrollArea, 'scr_area').verticalScrollBar()
        self.gbMask = self.findChild(QGroupBox, 'gb_mask')
        self.gbSeam = self.findChild(QGroupBox, 'gb_seam')
        self.worker = MainBackgroundThread()
        self.actOpen = self.findChild(QAction, 'act_open_2')
        self.actSave = self.findChild(QAction, 'act_save')
        self.actExit = self.findChild(QAction, 'act_exit')
        self.actSupport = self.findChild(QAction, 'act_support')
        self.actReport = self.findChild(QAction, 'act_report')
        self.actAbout = self.findChild(QAction, 'act_about')

        #attribute for objects and events
        self.btOpen.clicked.connect(self.OpenFile)
        self.btSave.clicked.connect(self.SaveFile)
        self.btRun.clicked.connect(self.RunAndStop)
        self.lbImg.mousePressEvent = self.CompareImg
        self.lbMask.mousePressEvent = self.mousePressEvent
        self.lbMask.mouseMoveEvent = self.mouseMoveEvent
        self.lbMask.mouseReleaseEvent = self.mouseReleaseEvent
        self.btClr.clicked.connect(self.ClearMask)
        self.btUndo.clicked.connect(self.UndoMask)
        self.ckbProtect.clicked.connect(self.ToggleProtect)
        self.ckbRemove.clicked.connect(self.ToggleRemove)
        self.ckbDownsize.clicked.connect(self.Downsize)
        self.sbSize.valueChanged.connect(lambda x: self.setBrushSize(x))
        self.scrBar.rangeChanged.connect(lambda x,y: self.scrBar.setValue(y))
        self.worker.finished.connect(self.onFinished)
        self.lbMask.setGeometry(self.lbImg.rect())
        self.lbMask.hide()
        self.actOpen.triggered.connect(self.OpenFile)
        self.actSave.triggered.connect(self.SaveFile)
        self.actExit.triggered.connect(lambda:self.close())
        self.actSupport.triggered.connect(lambda:webbrowser.open('https://github.com/zoxuandinhzo/seam-carving-with-pyqt5'))
        self.actReport.triggered.connect(lambda:webbrowser.open('https://github.com/zoxuandinhzo/seam-carving-with-pyqt5/issues'))
        self.actAbout.triggered.connect(lambda:self.About())

        #initialize variables
        self.runTime = 0
        self.img_in = None
        self.img_out = None
        self.imgShowing = None
        self.saveConfigInput = None
        self.listImask = []
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
            self.imgShowing = 'in'
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

    def RunAndStop(self):
        if self.btRun.text() == 'Run':
            self.RunSeam()
            self.btRun.setText('Stop')
        else:
            self.worker.stop()
            self.btRun.setText('Run')

    def RunSeam(self):
        if self.img_in is None:
            self.showInfor('No image to run.')
            return
        self.showInfor('START RUNNING !\nPLEASE WAIT !!!')
        dx = self.sbCols.value()
        dy = self.sbRows.value()
        downsize = self.ckbDownsize.isChecked()
        mode = self.cmbMode.currentText()
        vis = self.ckbProcess.isChecked()
        vismask = self.ckbMask.isChecked()
        vismap = self.ckbMap.isChecked()
        visall = self.ckbASeam.isChecked()
        if self.imgShowing == 'in':
            im = self.img_in
        else:
            im = self.img_out
            self.img_in = self.img_out
        mask = self.getMask(self.Imask, 255)
        rmask = self.getMask(self.Imask, 0)
        hremove = self.ckbHremove.isChecked()
        kratio = self.ckbRatio.isChecked()
        self.saveConfigInput = {'dx':dx, 'dy':dy, 'imask':self.Imask.copy(), 'hremove':hremove}
        self.worker.setting(im, dx, dy, mask, rmask, hremove, kratio, mode, vis, vismask, vismap, visall, downsize)
        self.runTime = time.time()
        self.worker.start()

    def CompareImg(self, event):
        if self.img_in is None or self.img_out is None:
            return
        if self.imgShowing == 'in':
            self.showImage(self.img_out)
            self.imgShowing = 'out'
        else:
            self.showImage(self.img_in)
            self.imgShowing = 'in'
            self.loadConfigInput(self.saveConfigInput)

    def loadConfigInput(self, config):
        self.sbCols.setValue(config['dx'])
        self.sbRows.setValue(config['dy'])
        self.Imask = config['imask'].copy()
        self.updateMask()
        self.ckbHremove.setChecked(config['hremove'])

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

    def Downsize(self):
        if self.ckbDownsize.isChecked():
            if self.imgShowing == 'in':
                h, w = self.img_in.shape[:2]
            else:
                h, w = self.img_out.shape[:2]
            if w < 500:
                self.showInfor("Your image don't need to be downsized")
                return
            dim = (int(h * 500 / float(w)), 500)
            self.showInfor('Your image will be downsized to ' + str(dim))

    def onFinished(self):
        self.btRun.setText('Run')
        self.img_out, all_seam = self.worker.getResult()
        if all_seam is not None:
            cv2.imshow('all seam', all_seam)
            cv2.waitKey(1)
            cv2.imwrite('all_seam.jpg', all_seam)
        if self.img_out is not None:
            self.showImage(self.img_out)
            self.imgShowing = 'out'
            self.runTime = time.time() - self.runTime
            self.showInfor('RUN DONE !\n- In size: {}\n- Out size: {}\n- Run time: {:.2f}s'.format(self.img_in.shape, self.img_out.shape, self.runTime))
        else:
            self.showInfor('The image is not changed !')

    def showImage(self, img):
        pixmap = ArrayToQPixmap(img).scaled(self.wgImg.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbImg.setPixmap(pixmap)
        self.lbImg.adjustSize()
        self.resetMask()

    def resetMask(self):
        self.sbCols.setValue(0)
        self.sbRows.setValue(0)
        self.ckbProtect.setCheckState(0)
        self.ckbRemove.setCheckState(0) 
        self.lbMask.hide()
        self.lbMask.resize(self.lbImg.size())
        self.Imask = self.Imask.scaled(self.lbImg.size())
        self.Imask.fill(self.bgColor)
        self.listImask = [self.Imask.copy()]
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
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
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
            self.listImask.append(self.Imask.copy())
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
        self.listImask.append(self.Imask.copy())
        self.updateMask()

    def UndoMask(self):
        if len(self.listImask)>1:
            self.listImask.pop()
            self.Imask=self.listImask[-1]
            self.updateMask()

    def updateMask(self):
        pixmap = QPixmap.fromImage(self.Imask)
        self.lbMask.setPixmap(pixmap)

    def setBrushSize(self, x):
        self.brushSize = x

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes |
            QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def resizeEvent(self, event):
        if self.img_in is not None:
            self.showImage(self.img_in)
        QMainWindow.resizeEvent(self, event)

    def About(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("About")
        dlg.setText("Seam Carving GUI with PyQt5 and Python\n" +
                    "-> Version: 1.0\n" +
                    "-> Lisence: None\n" + 
                    "-> Member: 18520603@gm.uit.edu.vn and 18520506@gm.uit.edu.vn\n" +
                    "-> Course: CS406.M11.KHCL\n" +
                    "-> University of Information Technology - UIT !!!")
        dlg.exec()

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