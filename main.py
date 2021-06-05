from robot import ILQRController, IdleController, ReferenceTrackerController
import sys
import enum

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

from diff_drive_robot import DifferentialDriveRobot, DifferentialDriveRobotFlatSystem
from diff_drive_robot_2 import DifferentialDriveRobot2, DifferentialDriveRobot2FlatSystem
from diff_drive_ellipse_wheel_robot import Ellipse, DifferentialDriveEllipseWheelRobot
from bicycle_robot import BicycleRobot, BicycleRobotFlatSystem
from bicycle_robot_2 import BicycleRobot2, BicycleRobot2FlatSystem

import SINDy_DiffDriveModel

from fsm_state import FsmState
from main_window import MainWindow
from sindy.sindy import sindy

class SindyUsage(enum.IntEnum):
    NONE = 0
    OFFLINE = 1
    ONLINE = 2
#/

def setup_diff_drive_robot(s0, sf, tf,
                           do_sindy=SindyUsage.NONE,
                           do_flatsys_init=False,
                           do_ilqr=False):
    robot = DifferentialDriveRobot(radius=15,
                                   wheel_radius=6,
                                   wheel_thickness=3)
    robot.reset(s0)

    # Choose robot model - actual or fit (SINDy)
    if do_sindy == SindyUsage.NONE:
        model = robot.model

    elif do_sindy == SindyUsage.OFFLINE:
        print('Loading pre-computed SINDy model')
        model = SINDy_DiffDriveModel.SINDy_DiffDriveModel()

    else: # do_sindy == SindyUsage.ONLINE
        print('Running SINDy')
        sindy_dyn_module_name = 'SINDy_DiffDriveModel'
        sindy(sindy_dyn_module_name,
              robot,
              n_control_samples=10000,
              n_state_samples_per_control=20,
              dt=tf/1000,
              threshold=1.0e-3,
              verbose=True)
        print('Loading in-situ SINDy model')
        module = __import__(sindy_dyn_module_name)
        model = module.SINDy_DiffDriveModel()
    # /if-else

    # Trajectory calculation - Diff Flat and/or iLQR
    N = 1000
    t = np.linspace(0, tf, N+1)
    if do_flatsys_init:
        print('Running flat-system trajectory generation')
        robot_flat = DifferentialDriveRobotFlatSystem(*robot.parameters())
        s_init, u_init = robot_flat.plan(s0, sf, t)
    else:
        u_init = np.zeros((N, model.controlDim()))
        s_init = model.generateTrajectory(t, s0, u_init)
    # /if-else use_flatsys_init
    
    if do_ilqr:
        print('Running iLQR')
        mat_Ls, vec_ls, t, s, u = \
            robot.ilqr(model, sf, t, s_init, u_init)
        assert len(t) == len(s) == len(u) # shapes must be equal for rendering
        controller = ILQRController(mat_Ls, vec_ls, t,s,u)
    elif do_flatsys_init:
        u_init = np.append([[0,0]], u_init, axis=0) # make shapes equal for rendering
        assert len(t) == len(s_init) == len(u_init)
        controller = ReferenceTrackerController(t, s_init, u_init)
    else:
        controller = IdleController(model.controlDim())
    # if-else

    robot.setController(controller)
    robot.drive()

    #/

    print('Done')
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
robot = setup_diff_drive_robot(s0, sf, tf,
                               do_sindy=SindyUsage.NONE,
                               do_flatsys_init=True,
                               do_ilqr=False)
# robot = setup_diff_drive_robot_2(s0, sf, tf)
# robot = setup_bicycle_robot(s0, sf, tf)
# robot = setup_bicycle_robot_2(s0, sf, tf)

app = QtWidgets.QApplication(sys.argv)
main_window = MainWindow(800, 800, robot)
main_window.show()
app.exec_()
