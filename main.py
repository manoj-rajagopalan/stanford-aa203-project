import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

from diff_drive_robot import DifferentialDriveRobot, DifferentialDriveRobotFlatSystem
from diff_drive_robot_2 import DifferentialDriveRobot2, DifferentialDriveRobot2FlatSystem
from diff_drive_ellipse_wheel_robot import Ellipse, DifferentialDriveEllipseWheelRobot
from bicycle_robot import BicycleRobot, BicycleRobotFlatSystem
from bicycle_robot_2 import BicycleRobot2, BicycleRobot2FlatSystem

from fsm_state import FsmState
from main_window import MainWindow
from sindy.sindy import sindy

def setup_diff_drive_robot(s0, sf, tf, use_sindy=False):
    robot = DifferentialDriveRobot(radius=15,
                                   wheel_radius=6,
                                   wheel_thickness=3)
    robot.reset(s0)
    if use_sindy:
        sindy_dyn_module_name = 'SINDy_DiffDriveModel'
        sindy(sindy_dyn_module_name,
              robot,
              n_control_samples=10000,
              n_state_samples_per_control=20,
              dt=tf/1000,
              threshold=1.0e-3,
              verbose=True)
        module = __import__(sindy_dyn_module_name)
        model = module.SINDy_DiffDriveModel()
    else:
        model = robot.model
    # /if-else

    # robot_flat = DifferentialDriveRobotFlatSystem(*robot.parameters())
    # t = np.linspace(0, tf, 1001)
    # s, u = robot_flat.plan(s0, sf, t)
    # robot.setTrajectory(t, s, u)
    
    robot.ilqr(model, sf, tf)
    
    return robot
 # /setup_diff_drive_robot()

def setup_diff_drive_robot_2(s0, sf, tf):
    robot = DifferentialDriveRobot2(radius=15,
                                    wheel_radius=6,
                                    wheel_thickness=3)
    s0 = np.append(s0,[0,0])
    sf = np.append(sf,[0,0])
    robot.reset(*s0)
    # robot_flat = DifferentialDriveRobotFlatSystem(*robot.parameters())
    # t = np.linspace(0, tf, 1001)
    # s, u = robot_flat.plan(s0, sf, t)
    # robot.setTrajectory(t, s, u)
    robot.gotoUsingIlqr(sf, tf)
    return robot
 # /setup_diff_drive_robot_2()

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
 # /setup_bicycle_robot()

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
 # /setup_bicycle_robot_2()

###############################################################################

# robot = DifferentialDriveEllipseWheelRobot(baseline=250,
#                                            left_wheel_ellipse=Ellipse(50, 10),
#                                            right_wheel_ellipse=Ellipse(50, 10),
#                                            wheel_thickness=5)
# robot.reset(250, 270, 0, 0, 90)


s0 = np.array([40, 40, 0])
sf = np.array([600, 300, np.deg2rad(179)])
tf = 10 # s
robot = setup_diff_drive_robot(s0, sf, tf)
# robot = setup_diff_drive_robot_2(s0, sf, tf)
# robot = setup_bicycle_robot(s0, sf, tf)
# robot = setup_bicycle_robot_2(s0, sf, tf)

app = QtWidgets.QApplication(sys.argv)
main_window = MainWindow(800, 800, robot)
main_window.show()
app.exec_()
