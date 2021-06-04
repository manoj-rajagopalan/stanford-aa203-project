import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

from diff_drive_robot import DifferentialDriveRobot, DifferentialDriveRobotFlatSystem
from diff_drive_robot_2 import DifferentialDriveRobot2, DifferentialDriveRobot2FlatSystem
from diff_drive_ellipse_wheel_robot import Ellipse, DifferentialDriveEllipseWheelRobot
from bicycle_robot import BicycleRobot, BicycleRobotFlatSystem
from bicycle_robot_2 import BicycleRobot2, BicycleRobot2FlatSystem

from fsm_state import FsmState
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, width, height, robot):
        super(MainWindow, self).__init__()
        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(width, height)
        self.label.setPixmap(canvas)
        self.setCentralWidget(self.label)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.render)
        self.counter = 0
        self.robot = robot
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
        if self.robot.fsm_state == FsmState.IDLE:
            self.robot.drive()
        # /if

        # screen coords (top left, downwards) -> math coords (bottom left, upwards)
        qpainter.translate(0, self.height()-1)
        qpainter.scale(1, -1)

        self.robot.render(qpainter)

        qpainter.end()
        self.update()
    # /render()

# /class MainWindow

app = QtWidgets.QApplication(sys.argv)

def setup_diff_drive_robot(s0, sf, tf):
    robot = DifferentialDriveRobot(radius=15,
                                   wheel_radius=6,
                                   wheel_thickness=3)
    robot.reset(*s0)
    robot_flat = DifferentialDriveRobotFlatSystem(*robot.parameters())
    t = np.linspace(0, tf, 1001)
    s, u = robot_flat.plan(s0, sf, t)
    robot.setTrajectory(t, s, u)
    # robot.gotoUsingIlqr(sf, tf)
    return robot
 # /setup_diff_drive_robot()

# robot = DifferentialDriveEllipseWheelRobot(baseline=250,
#                                            left_wheel_ellipse=Ellipse(50, 10),
#                                            right_wheel_ellipse=Ellipse(50, 10),
#                                            wheel_thickness=5)
# robot.reset(250, 270, 0, 0, 90)

def setup_bicycle_robot(s0, sf, tf):
    robot = BicycleRobot(wheel_radius=20,
                         baseline=60)
    robot.reset(*s0)
    # robot_flat = BicycleRobotFlatSystem(*robot.parameters())
    # t = np.linspace(0, tf, 1001)
    # s, u = robot_flat.plan(s0, sf, t)
    # robot.setTrajectory(t, s, u)
    robot.gotoUsingIlqr(sf, tf)
    return robot
 # /setup_diff_drive_robot()

def setup_bicycle_robot_2(s0, sf, tf):
    robot = BicycleRobot2(wheel_radius=20,
                          baseline=60)
    s0 = np.append(s0,0)
    sf = np.append(sf,0)
    robot.reset(*s0)
    # robot_flat = BicycleRobot2FlatSystem(*robot.parameters())
    # t = np.linspace(0, tf, 1001)
    # s, u = robot_flat.plan(s0, sf, t)
    # robot.setTrajectory(t, s, u)
    robot.gotoUsingIlqr(sf, tf)
    return robot
 # /setup_diff_drive_robot()

# robot = BicycleRobot2(wheel_radius=20, baseline=60)
# s0 = np.array([40,40,0, 0])
# sf = np.array([100,100,179, 0])
# robot.reset(*s0)
# t = np.linspace(0,10,101)
# robot_flat = BicycleRobot2FlatSystem(robot.r, robot.L)
# s, u = robot_flat.plan(s0, sf, t)
# s, u = s.T, u.T
# robot.setTrajectory(t, s, u[:-1])
# robot.gotoUsingIlqr(sf, 10)

# robot = DifferentialDriveRobot2(radius=15, wheel_radius=6, wheel_thickness=3)
# s0 = np.array([40,40,0, 0,0])
# sf = np.array([600,300,179, 0,0])
# robot.reset(*s0)
# t = np.linspace(0,10,101)
# robot_flat = DifferentialDriveRobot2FlatSystem(robot.r, 2*robot.R)
# s, u = robot_flat.plan(s0, sf, t, robot.controlLimits())
# s, u = s.T, u.T
# robot.setTrajectory(t, s, u[:-1])
# robot.gotoUsingIlqr(sf, 5)

s0 = np.array([40, 40, 0])
sf = np.array([600, 300, -179])
tf = 10 # s
# robot = setup_diff_drive_robot(s0, sf, tf)
# robot = setup_bicycle_robot(s0, sf, tf)
robot = setup_bicycle_robot_2(s0, sf, tf)
window = MainWindow(800, 600, robot)
window.show()

app.exec_()
