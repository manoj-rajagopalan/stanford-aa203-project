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

        t = np.arange(0, 5, 0.1)
        omega_l = 3.0
        omega_r = 4.0
        self.states = np.empty((len(t), 3))
        self.states[0] = diff_dr_robot.state
        for i in range(1,len(t)):
            dt = t[i] - t[i-1]
            if t[i] > 2.5:
                omega_l, omega_r = omega_r, omega_l
            diff_dr_robot.applyControl(omega_l, omega_r, dt)
            self.states[i] = diff_dr_robot.state
        # /for i
        
        self.diff_dr_robot = diff_dr_robot
        self.timer.start(int(dt * 1000))

    # /__init__()

    def animate(self):
        if self.counter < self.states.shape[0]:
            self.diff_dr_robot.state = self.states[self.counter]
            self.render()
            # print('Rendered state ', self.counter, ': ', self.states[self.counter])
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
                                                        wheel_radius=20,
                                                        wheel_thickness=10)
window = MainWindow(600, 400, diff_dr_robot)
window.show()
app.exec_()
