import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

import diff_drive_robot

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, width, height, diff_dr_robot):
        super(MainWindow, self).__init__()
        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(width, height)
        self.label.setPixmap(canvas)
        self.setCentralWidget(self.label)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.animate)
        self.counter = 0

        t = np.arange(0, width, 10)
        self.trajectory = np.empty((len(t), 3))
        self.trajectory[:,0] = t
        self.trajectory[:,1] = (height / 2) + 50 * np.sin(2 * np.pi * (t / width))
        deriv = 50 * np.cos(2 * np.pi * (t / width)) * (2 * np.pi) / width
        self.trajectory[:,2] = np.arctan2(deriv, np.ones(len(t)))
        
        self.diff_dr_robot = diff_dr_robot

        self.timer.start(100)

    # /__init__()

    def animate(self):
        if self.counter < self.trajectory.shape[0]:
            # palette = self.label.palette()
            # palette.setColor(self.backgroundRole(), QtCore.Qt.white)
            # self.label.setPalette(palette)
            self.diff_dr_robot.setPose(*self.trajectory[self.counter,:])
            self.render()
            # print('Rendered frame', self.counter, 'with angle', self.trajectory[self.counter,2] * 180/np.pi)
            self.counter += 1
            self.update()
    # /animate()

    def render(self):
        qpainter = QtGui.QPainter(self.label.pixmap())

        # background
        bg_brush = QtGui.QBrush()
        bg_brush.setColor(QtCore.Qt.white)
        bg_brush.setStyle(QtCore.Qt.SolidPattern)
        qpainter.setBrush(bg_brush)
        qpainter.drawRect(0,0, self.width(), self.height())

        # foreground
        self.diff_dr_robot.render(qpainter, self.label.height())

        qpainter.end()
    # /render()

# /class MainWindow

app = QtWidgets.QApplication(sys.argv)
diff_dr_robot = diff_drive_robot.DifferentialDriveRobot(radius=50,
                                                        wheel_radius=40,
                                                        wheel_thickness=5)
window = MainWindow(400, 300, diff_dr_robot)
window.show()
app.exec_()
