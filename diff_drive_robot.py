from models.diff_drive_model import DiffDriveModel
import time
import copy

import numpy as np
import scipy.integrate
import scipy.optimize
import control.flatsys as flatsys
from PyQt5 import QtGui, QtCore

from robot import Robot
from fsm_state import FsmState
from ilqr import iLQR
class DifferentialDriveRobotFlatSystem(flatsys.FlatSystem):
    def __init__(self, r, L):
        self.r = r # wheel radius
        self.L = L # baseline
        super(DifferentialDriveRobotFlatSystem, self).__init__(self.forward,
                                                               self.reverse,
                                                               params={'r': r, 'L': L},
                                                               inputs=['ω_l', 'ω_r'],
                                                               outputs=['flat_x', 'flat_y'],
                                                               states=['x', 'y', 'θ'])
    # /__init__()

    def forward(self, s, u):
        r, L = self.r, self.L
        x, y, θ = s
        ω_l, ω_r = u
        v = r * 0.5 * (ω_l + ω_r)
        x_dot = v * np.cos(θ)
        y_dot = v * np.sin(θ)
        θ_dot = (r / L) * (ω_r - ω_l)
        x_ddot = -y_dot * θ_dot
        y_ddot = x_dot * θ_dot
        # 'control' package calls this the 'flag'
        flat_flag = [np.array([x, x_dot, x_ddot]),
                     np.array([y, y_dot, y_ddot])]
        return flat_flag
    # /forward()

    def reverse(self, flat_flag):
        r, L = self.r, self.L
        x, x_dot, x_ddot = flat_flag[0]
        y, y_dot, y_ddot = flat_flag[1]
        θ = np.arctan2(y_dot, x_dot)
        v_sqr = (x_dot * x_dot) + (y_dot * y_dot)
        v = np.sqrt(v_sqr)
        ωr_plus_ωl = (2/r) * v
        ωr_minus_ωl = (L/r) * (x_dot * y_ddot - x_ddot * y_dot) / v_sqr
        ωr = 0.5 * (ωr_plus_ωl + ωr_minus_ωl)
        ωl = ωr_plus_ωl - ωr
        s = np.array([x, y, θ])
        u = np.array([ωl, ωr])
        return s, u
    # /reverse()

    def plan(self, s0, sf, timepts):
        N = len(timepts)
        x0 = copy.deepcopy(s0)
        x0[2] = np.deg2rad(s0[2])
        xf = copy.deepcopy(sf)
        xf[2] = np.deg2rad(sf[2])
        constraints = None
        # constraint_A = [np.diag([0,0,0,1,1])] * N
        # constraint_lb = [np.array([0,0,0,-5,-5])] * N
        # constraint_ub = [np.array([0,0,0, 5, 5])] * N
        # constraints = [(scipy.optimize.LinearConstraint, constraint_A, constraint_lb, constraint_ub)]
        # control_extractor = lambda x,u : u
        # constraint_lb = [np.array([-5,-5])] * N
        # constraint_ub = [np.array([ 5, 5])] * N
        # constraints = [(scipy.optimize.NonlinearConstraint, control_extractor, constraint_lb, constraint_ub)]
        cost = None
        # cost = lambda x, u : np.dot(x - target_state, x - target_state) + np.dot(u,u)
        basis = None
        # basis = flatsys.PolyFamily(8)
        traj_func = flatsys.point_to_point(self, timepts, x0=x0, xf=xf, constraints=constraints, cost=cost, basis=basis)
        s, u = traj_func.eval(timepts)
        return s.T, u.T[:-1]
    # /plan()

# /class DifferentialDriveRobotFlatSystem

class DifferentialDriveRobot(Robot):
    
    def __init__(self, radius, wheel_radius, wheel_thickness):
        model = DiffDriveModel(wheel_radius, 2 * radius)
        super(DifferentialDriveRobot, self).__init__(model)

        self.radius = radius
        self.wheel_radius = wheel_radius
        self.wheel_thickness = wheel_thickness
    # /__init__()

    def reset(self, x, y, θ_deg):
        self.s = np.array([[x, y, np.deg2rad(θ_deg)]])
    # /

    def controlLimits(self):
        u_max = np.array([30.0, 30.0]) # rad/s
        u_min = -u_max
        return u_min, u_max
    # /controlLimits()

    def ilqr(self, model, s_goal, duration, dt=0.01):
        self.fsmTransition(FsmState.PLANNING)
        N = int(duration / dt)
        P_N = 500 * np.eye(model.stateDim())
        Q = np.array([np.diag([1,1,1])] * N)
        # Q = np.eye(3) + (np.arange(N)/N)[:, np.newaxis, np.newaxis] * 0.01*P_N[np.newaxis, :, :]
        R_k = 5 * np.eye(model.controlDim())
        R_delta_u = 1000 * np.eye(model.controlDim())
        s, u, mat_Ls, vec_ls, metrics_history = \
            iLQR(model,
                 self.s[-1], s_goal, N, dt,
                 P_N, Q, R_k, R_delta_u, 100)
        t = np.linspace(0,N,N+1) * dt
        self.setTrajectory(t, s, u)
        self.drive()
    # /gotoUsingIlqr()

    # Draw this instance onto a qpainter
    def renderCanonical(self, qpainter):
        '''
            Renders the robot in canonical coordinate frame.
            Call after setting the world transformation.
        '''
        # main body
        brush = QtGui.QBrush()
        brush.setColor(QtCore.Qt.black)
        brush.setStyle(QtCore.Qt.SolidPattern)
        qpainter.setBrush(brush)
        qpainter.drawEllipse(QtCore.QPoint(0, 0), self.radius, self.radius)

        # ... single dot to mark orientation
        brush.setColor(QtCore.Qt.yellow)
        qpainter.setBrush(brush)
        qpainter.drawEllipse(QtCore.QPoint(0.75 * self.radius, 0), 0.1 * self.radius, 0.1 * self.radius)

        # wheels
        brush.setColor(QtCore.Qt.red)
        qpainter.setBrush(brush)
        qpainter.drawRect(-self.wheel_radius, -self.radius -self.wheel_thickness, 2 * self.wheel_radius, self.wheel_thickness)
        qpainter.drawRect(-self.wheel_radius,  self.radius,                       2 * self.wheel_radius, self.wheel_thickness)

    # /renderCanonical()

    def plotTrajectory(self, state_plot, control_plot):
        state_plot.distance_axes.set_ylabel('$x$, $y$')
        state_plot.distance_axes.plot(self.t, self.s[:,0], 'r', label='$x$')
        state_plot.distance_axes.plot(self.t, self.s[:,1], 'g', label='$y$')

        state_plot.angle_axes.set_ylabel('$\\theta$ (deg)')
        state_plot.angle_axes.plot(self.t, np.rad2deg(self.s[:,2]), 'b', label='$\theta$')

        control_plot.angle_axes.set_ylabel('$\\omega_l$, $\\omega_r$ (deg/s)')
        control_plot.angle_axes.plot(self.t[:-1], np.rad2deg(self.u[:,0]), 'r', label='$\\omega_l$')
        control_plot.angle_axes.plot(self.t[:-1], np.rad2deg(self.u[:,1]), 'g', label='$\\omega_r$')
    # /plotTrajectory()
# /class DifferentialDriveRobot
