from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

class Window(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        layout = QtWidgets.QGridLayout(self)

        self.getImageButton = QtWidgets.QPushButton('Select')
        layout.addWidget(self.getImageButton)
        self.getImageButton.clicked.connect(self.resimac)

        self.resim1 = QtWidgets.QLabel()
        layout.addWidget(self.resim1)
        self.resim1.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        # I'm assuming the following...
        self.resim1.setScaledContents(True)
        self.resim1.setFixedSize(701,451)

        # install an event filter to "capture" mouse events (amongst others)
        self.resim1.installEventFilter(self)

    def resimac(self):
        filename, filter = QtWidgets.QFileDialog.getOpenFileName(None, 'Resim Yükle', '.', 'Image Files (*.png *.jpg *.jpeg *.bmp *.tif)')
        if not filename:
            return
        self.resim1.setPixmap(QtGui.QPixmap(filename).scaled(self.resim1.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def eventFilter(self, source, event):
        # if the source is our QLabel, it has a valid pixmap, and the event is
        # a left click, proceed in trying to get the event position
        if (source == self.resim1 and source.pixmap() and not source.pixmap().isNull() and 
            event.type() == QtCore.QEvent.MouseButtonPress and
            event.button() == QtCore.Qt.LeftButton):
                self.getClickedPosition(event.pos())
        return super().eventFilter(source, event)

    def getClickedPosition(self, pos):
        # consider the widget contents margins
        contentsRect = QtCore.QRectF(self.resim1.contentsRect())
        if pos not in contentsRect:
            # outside widget margins, ignore!
            return

        # adjust the position to the contents margins
        pos -= contentsRect.topLeft()

        pixmapRect = self.resim1.pixmap().rect()
        if self.resim1.hasScaledContents():
            x = pos.x() * pixmapRect.width() / contentsRect.width()
            y = pos.y() * pixmapRect.height() / contentsRect.height()
            pos = QtCore.QPoint(x, y)
        else:
            align = self.resim1.alignment()
            # for historical reasons, QRect (which is based on integer values),
            # returns right() as (left+width-1) and bottom as (top+height-1),
            # and so their opposite functions set/moveRight and set/moveBottom
            # take that into consideration; using a QRectF can prevent that; see:
            # https://doc.qt.io/qt-5/qrect.html#right
            # https://doc.qt.io/qt-5/qrect.html#bottom
            pixmapRect = QtCore.QRectF(pixmapRect)

            # the pixmap is not left aligned, align it correctly
            if align & QtCore.Qt.AlignRight:
                pixmapRect.moveRight(contentsRect.x() + contentsRect.width())
            elif align & QtCore.Qt.AlignHCenter:
                pixmapRect.moveLeft(contentsRect.center().x() - pixmapRect.width() / 2)
            # the pixmap is not top aligned (note that the default for QLabel is
            # Qt.AlignVCenter, the vertical center)
            if align & QtCore.Qt.AlignBottom:
                pixmapRect.moveBottom(contentsRect.y() + contentsRect.height())
            elif align & QtCore.Qt.AlignVCenter:
                pixmapRect.moveTop(contentsRect.center().y() - pixmapRect.height() / 2)

            if not pos in pixmapRect:
                # outside image margins, ignore!
                return
            # translate coordinates to the image position and convert it back to
            # a QPoint, which is integer based
            pos = (pos - pixmapRect.topLeft()).toPoint()

        print('X={}, Y={}'.format(pos.x(), pos.y()))


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())