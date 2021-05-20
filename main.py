import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

import diff_drive_robot
from diff_drive_ellipse_wheel_robot import Ellipse, DifferentialDriveEllipseWheelRobot

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, width, height, diff_dr_robot):
        super(MainWindow, self).__init__()
        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(width, height)
        self.label.setPixmap(canvas)
        self.setCentralWidget(self.label)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.render)
        self.counter = 0

        if False:
            t = np.arange(0, 5, 0.1)
            omega_l = 3.0
            omega_r = 4.0
            self.states = np.empty((len(t), 3))
            self.states[0] = diff_dr_robot.state
            for i in range(1,len(t)):
                dt = t[i] - t[i-1]
                if t[i] > 2.5:
                    omega_l, omega_r = omega_r, omega_l
                u = np.array([omega_l, omega_r])
                diff_dr_robot.applyControl(dt, u)
                self.states[i] = diff_dr_robot.state
            # /for i
        # /if

        self.diff_dr_robot = diff_dr_robot
        self.timer.start(100) # ms

    # /__init__()

    # def animate(self):
    #     if self.counter < self.states.shape[0]:
    #         self.diff_dr_robot.state = self.states[self.counter]
    #         self.render()
    #         # print('Rendered state ', self.counter, ': ', self.states[self.counter])
    #         self.counter += 1
    #         self.update()
    # # /animate()

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
        self.update()
    # /render()

# /class MainWindow

app = QtWidgets.QApplication(sys.argv)
# diff_dr_robot = diff_drive_robot.DifferentialDriveRobot(radius=15,
#                                                         wheel_radius=6,
#                                                         wheel_thickness=3)
diff_dr_robot = DifferentialDriveEllipseWheelRobot(baseline=250,
                                                   left_wheel_ellipse=Ellipse(50, 10, 0),
                                                   right_wheel_ellipse=Ellipse(50, 10, 90),
                                                   wheel_thickness=5,
                                                   initial_xyθ=np.array([250, 270, 0]))
window = MainWindow(800, 600, diff_dr_robot)
window.show()
# diff_dr_robot.goto(np.array([500, 300, np.pi/180 * 90]), 5.0)
diff_dr_robot.plotHistory(True)
diff_dr_robot.go(np.array([[60,60]]) * np.pi/180, np.array([15]))
app.exec_()
