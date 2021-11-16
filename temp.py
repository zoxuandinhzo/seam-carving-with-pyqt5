from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys, os

class myQLabel(QLabel):
    def __init__(self,parent=None):
        super(myQLabel, self).__init__(parent)

    def paintEvent(self, QPaintEvent):
        super(myQLabel, self).paintEvent(QPaintEvent)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red))
        painter.drawArc(QRectF(50, 50, 10, 10), 0, 5760)
        painter.drawRect(QRectF(50, 50, 100, 100) )

class Example(QWidget):

    def __init__(self,parent):
        super(Example, self).__init__()
        self.main_image_name="icons\\add.png"
        self.mode = 5

        self.initUI()

    def initUI(self):
        File_name = QLabel('Setup file name')
        File_name_edit = QLineEdit()
        QToolTip.setFont(QFont('SansSerif', 10))
        #QMainWindow.statusBar().showMessage('Ready')
        self.setGeometry(300, 300, 250, 150)
        self.resize(640, 360)
        self.center()
        self.main_image = myQLabel(self)
        self.main_image.setPixmap(QPixmap(self.main_image_name))
        btn = QPushButton("Make setup file")
        btn.setToolTip('Press <b>Detect</b> button for detecting objects by your settings')
        btn.resize(btn.sizeHint())
        btn.clicked.connect(QCoreApplication.instance().quit)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse)
        btn_set = QPushButton("Set name")
        #fullscreen
        #self.main_image.setScaledContents(True)
        #just centered
        self.main_image.setAlignment(Qt.AlignCenter)



        #Layout
        box_File_name = QHBoxLayout()
        box_File_name.addWidget(File_name)
        box_File_name.addWidget(File_name_edit)
        box_File_name.addWidget(btn_set)
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addLayout(box_File_name, 1, 0)
        #grid.addWidget(File_name_edit, 1, 1)
        grid.addWidget(self.main_image, 2, 0)
        grid.addWidget(btn_browse, 3 , 0)
        grid.addWidget(btn, 4, 0)

        box_number = QVBoxLayout()
        number_group=QButtonGroup() # Number group
        r0=QRadioButton("Traffic Lights")
        number_group.addButton(r0)
        r1=QRadioButton("Direction")
        number_group.addButton(r1)
        r2=QRadioButton("Traffic Lines H")
        number_group.addButton(r2)
        r3=QRadioButton("Traffic Lines V")
        number_group.addButton(r3)
        box_number.addWidget(r0)
        box_number.addWidget(r1)
        box_number.addWidget(r2)
        box_number.addWidget(r3)

        r0.toggled.connect(self.radio0_clicked)
        r1.toggled.connect(self.radio1_clicked)
        r2.toggled.connect(self.radio2_clicked)
        r3.toggled.connect(self.radio3_clicked)

        box_road_sign = QHBoxLayout()
        road_sign_label = QLabel('Road signs', self)
        road_sign = QComboBox()
        road_sign.addItem("None")
        road_sign.addItem("ex1")
        road_sign.addItem("ex2")
        road_sign.addItem("ex3")
        road_sign.addItem("ex4")
        road_sign.addItem("ex5")
        box_road_sign.addWidget(road_sign_label)
        box_road_sign.addWidget(road_sign)
        grid.addLayout(box_road_sign, 1, 1)
        grid.addLayout(box_number, 2, 1)
        self.setLayout(grid)

        self.show()
    def browse(self):
        w = QWidget()
        w.resize(320, 240)
        w.setWindowTitle("Select Picture")
        filename = QFileDialog.getOpenFileName(w, 'Open File', '/')
        self.main_image_name = filename
        self.main_image.setPixmap(QPixmap(self.main_image_name))

    def center(self):

        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


    def radio0_clicked(self, enabled):
        if enabled:
            print("0")
            self.mode=0

    def radio1_clicked(self, enabled):
        if enabled:
            print("1")
            self.mode=1

    def radio2_clicked(self, enabled):
        if enabled:
            print("2")
            self.mode=2

    def radio3_clicked(self, enabled):
        if enabled:
            print("3")
            self.mode=3

    # def paintEvent( self, event) :
    #     pass

class menubarex(QMainWindow):
    def __init__(self, parent=None):
        super(menubarex, self).__init__(parent)
        self.form_widget = Example(self)
        self.setCentralWidget(self.form_widget)

        self.initUI()

    def initUI(self):
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

        #self.toolbar = self.addToolBar('Exit')
        #self.toolbar.addAction(exitAction)

        self.statusBar().showMessage('Ready')
        self.setWindowTitle('mi ban')
        self.setWindowIcon(QIcon('icon.png'))

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes |
            QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

def main():
    app = QApplication(sys.argv)
    #ex = Example()
    menubar = menubarex()
    menubar.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()