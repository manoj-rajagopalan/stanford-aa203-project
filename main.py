import typing
from robot import ILQRController, IdleController, ReferenceTrackerController
import sys
import enum

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import pyplot as plt

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

def displayIlqrMetrics(metrics):
    fig, axes = plt.subplots(1,3)
    fig.suptitle('iLQR Metrics')
    [ax.set_xlabel('Iteration') for ax in axes]
    axes[0].set_ylabel('Cost')
    axes[0].set_yscale('log')
    axes[0].plot(metrics['cost'])
    axes[1].set_ylabel('$||\mathbf{s}_f - \mathbf{s}_{goal}||_2$')
    axes[1].set_yscale('log')
    axes[1].plot(metrics['sf_norm'])
    axes[2].set_ylabel('$||\Delta \mathbf{u}||_\infty$')
    axes[2].plot(metrics['du_norm'])
    axes[2].set_yscale('log')
    plt.show()
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
    dt = 0.005
    N = int(tf / dt)
    t = np.linspace(0, tf, N+1)
    if do_flatsys_init:
        print('Running flat-system trajectory generation')
        robot_flat = DifferentialDriveRobotFlatSystem(*robot.parameters())
        s_init, u_init = robot_flat.previewplan(s0, sf, t)
    else:
        u_init = np.zeros((N, model.controlDim()))
        s_init = model.generateTrajectory(t, s0, u_init)
    # /if-else use_flatsys_init
    
    ilqr_metrics = None

    if do_ilqr:
        print('Running iLQR')
        mat_Ls, vec_ls, t, s, u, ilqr_metrics = \
            robot.ilqr(model, sf, t, s_init, u_init)
        assert len(t) == len(s) == len(u) # shapes must be equal for rendering
        # ilqr_metrics_display_thread = IlqrMetricsDisplay(ilqr_metrics_history)
        # ilqr_metrics_display_thread.start()
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
    return robot, ilqr_metrics
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

app = QtWidgets.QApplication(sys.argv)

s0 = np.array([40, 40, 0])
sf = np.array([600, 300, np.deg2rad(135)])
tf = 10 # s
robot, ilqr_metrics = \
    setup_diff_drive_robot(s0, sf, tf,
                           do_sindy=SindyUsage.NONE,
                           do_flatsys_init=False,
                           do_ilqr=True)
# robot = setup_diff_drive_robot_2(s0, sf, tf)
# robot = setup_bicycle_robot(s0, sf, tf)
# robot = setup_bicycle_robot_2(s0, sf, tf)

main_window = MainWindow(800, 800, robot)
main_window.setGoalPose(sf)
main_window.show()
app.exec_()

if ilqr_metrics is not None:
    displayIlqrMetrics(ilqr_metrics)
#/
