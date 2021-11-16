#!/usr/bin/env python3

import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg

#!/usr/bin/env python3

import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg
class TestImage(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        # Basic UI layout
        self.statusbar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.statusbar)
        self.glw = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.glw)

        # Make image plot
        self.p1 = self.glw.addPlot()
        self.p1.getViewBox().setAspectLocked()
        # Draw axes and ticks above image/data
        [ self.p1.getAxis(ax).setZValue(10) for ax in self.p1.axes ]
        self.data = np.random.rand(120, 100)
        self.img = pg.ImageItem(self.data)
        self.p1.addItem(self.img)
        # Centre axis ticks on pixel
        self.img.setPos(-0.5, -0.5)

        # Swap commented lines to choose between hover or click events
        self.p1.scene().sigMouseMoved.connect(self.mouseMovedEvent)
        #self.p1.scene().sigMouseClicked.connect(self.mouseClickedEvent)

    def mouseClickedEvent(self, event):
        self.mouseMovedEvent(event.pos())

    def mouseMovedEvent(self, pos):
        # Check if event is inside image, and convert from screen/pixels to image xy indicies
        if self.p1.sceneBoundingRect().contains(pos):
            mousePoint = self.p1.getViewBox().mapSceneToView(pos)
            x_i = round(mousePoint.x())
            y_i = round(mousePoint.y())
            if x_i > 0 and x_i < self.data.shape[0] and y_i > 0 and y_i < self.data.shape[1]:
                self.statusbar.showMessage("({}, {}) = {:0.2f}".format(x_i, y_i, self.data[x_i, y_i]))
                return
        self.statusbar.clearMessage()

def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = TestImage()
    mainwindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()