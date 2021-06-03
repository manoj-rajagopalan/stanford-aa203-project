import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

from diff_drive_robot import DifferentialDriveRobot
from diff_drive_ellipse_wheel_robot import Ellipse, DifferentialDriveEllipseWheelRobot
from bicycle_robot import BicycleRobot, BicycleRobotFlatSystem
from bicycle_robot_2 import BicycleRobot2, BicycleRobot2FlatSystem

from fsm_state import FsmState
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
        self.diff_dr_robot = diff_dr_robot
        self.timer.start(25) # ms

    # /__init__()

    def render(self):
        qpainter = QtGui.QPainter(self.label.pixmap())

        # background
        bg_brush = QtGui.QBrush()
        bg_brush.setColor(QtCore.Qt.white)
        bg_brush.setStyle(QtCore.Qt.SolidPattern)
        qpainter.setBrush(bg_brush)
        qpainter.drawRect(0,0, self.width(), self.height())

        # foreground
        if self.diff_dr_robot.fsm_state == FsmState.IDLE:
            self.diff_dr_robot.drive()
        # /if

        # screen coords (top left, downwards) -> math coords (bottom left, upwards)
        qpainter.translate(0, self.height()-1)
        qpainter.scale(1, -1)

        self.diff_dr_robot.render(qpainter, self.label.height())

        qpainter.end()
        self.update()
    # /render()

# /class MainWindow

app = QtWidgets.QApplication(sys.argv)

# robot = diff_drive_robot.DifferentialDriveRobot(radius=15,
#                                                 wheel_radius=6,
#                                                 wheel_thickness=3)
# robot.reset(20, 40, 0)

# robot = DifferentialDriveEllipseWheelRobot(baseline=250,
#                                            left_wheel_ellipse=Ellipse(50, 10),
#                                            right_wheel_ellipse=Ellipse(50, 10),
#                                            wheel_thickness=5)
# robot.reset(250, 270, 0, 0, 90)

# robot = BicycleRobot(wheel_radius=20, baseline=60)
# s0 = np.array([40,40,0])
# robot.reset(*s0)
# robot_flat = BicycleRobotFlatSystem(robot.r, robot.L)
# t = np.linspace(0,10,1001)
# s, u = robot_flat.plan(s0,
#                        sf=np.array([600,300,90]),
#                        timepts=t)
# s, u = s.T, u.T
# robot.setTrajectory(s, u[:-1], t)
# robot.gotoUsingIlqr(np.array([600,300,90]), 10)

robot = BicycleRobot2(wheel_radius=20, baseline=60)
s0 = np.array([40,40,0, 0])
sf = np.array([100,100,179, 0])
robot.reset(*s0)
# t = np.linspace(0,10,101)
# robot_flat = BicycleRobot2FlatSystem(robot.r, robot.L)
# s, u = robot_flat.plan(s0,
#                        sf=sf,
#                        timepts=t)
# s, u = s.T, u.T
# robot.setTrajectory(s, u[:-1], t)
robot.gotoUsingIlqr(sf, 10)


window = MainWindow(800, 600, robot)
window.show()
# robot.goto(np.array([600, 300, np.pi/180 * 179]), 5.0)
# robot.plotHistory(True)
# robot.go(np.array([[60,60]]) * np.pi/180, np.array([15]))

app.exec_()
