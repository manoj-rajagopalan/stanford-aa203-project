import time
import copy

import numpy as np
import scipy.integrate
import scipy.optimize
import control.flatsys as flatsys
from PyQt5 import QtGui, QtCore

from fsm_state import FsmState
from robot import Robot
from ilqr import iLQR

class DifferentialDriveRobot2FlatSystem(flatsys.FlatSystem):
    def __init__(self, r, L):
        self.r = r # wheel radius
        self.L = L # baseline
        super(DifferentialDriveRobot2FlatSystem, self).__init__(self.forward,
                                                                self.reverse,
                                                                inputs=['α_l', 'α_r'],
                                                                outputs=['flat_x', 'flat_y'],
                                                                states=['x', 'y', 'θ', 'ω_l', 'ω_r'])
    # /__init__()

    def forward(self, s, u):
        r, L = self.r, self.L
        x, y, θ, ω_l, ω_r = s
        α_l, α_r = u
        v = r * 0.5 * (ω_l + ω_r)
        x_dot = v * np.cos(θ)
        y_dot = v * np.sin(θ)
        θ_dot = (r / L) * (ω_r - ω_l)
        ω_l_dot = α_l
        ω_r_dot = α_r
        x_ddot = -y_dot * θ_dot
        y_ddot =  x_dot * θ_dot
        θ_ddot = (r/L) * (ω_r_dot - ω_l_dot)
        x_dddot = -y_ddot * θ_dot - y_dot * θ_ddot
        y_dddot =  x_ddot * θ_dot + x_dot * θ_ddot
        # 'control' package calls this the 'flag'
        flat_flag = [np.array([x, x_dot, x_ddot, x_dddot]),
                     np.array([y, y_dot, y_ddot, y_dddot])]
        return flat_flag
    # /forward()

    def reverse(self, flat_flag):
        r, L = self.r, self.L
        x, x_dot, x_ddot, x_dddot = flat_flag[0]
        y, y_dot, y_ddot, y_dddot = flat_flag[1]
        θ = np.arctan2(y_dot, x_dot)
        v_sqr = (x_dot * x_dot) + (y_dot * y_dot)
        v = np.sqrt(v_sqr)
        ωr_plus_ωl = (2/r) * v
        xymxy = x_dot * y_ddot - x_ddot * y_dot
        ωr_minus_ωl = (L/r) * xymxy / v_sqr
        ωr = 0.5 * (ωr_plus_ωl + ωr_minus_ωl)
        ωl = ωr_plus_ωl - ωr
        xxpyy = x_dot * x_ddot + y_dot * y_ddot
        αr_plus_αl = 2/(r * v) * xxpyy
        αr_minus_αl = L/(r * v_sqr) * ((x_dot*y_dddot - y_dot*x_dddot) - 2 *xymxy * xxpyy / v_sqr)
        αr = 0.5 * (αr_plus_αl + αr_minus_αl)
        αl = αr_plus_αl - αr
        s = np.array([x, y, θ, ωl, ωr])
        u = np.array([αl,αr])
        return s, u
    # /reverse()

    def plan(self, s0, sf, t, u_limits=None):
        x0 = copy.deepcopy(s0)
        x0[2:] = np.deg2rad(s0[2:])
        xf = copy.deepcopy(sf)
        xf[2:] = np.deg2rad(sf[2:])
        constraints = None
        if u_limits is not None:
            constraint_lb = np.array([-5,-5]) # np.array([0,0,0,-5,-5])
            constraint_ub = np.array([ 5, 5]) # np.array([0,0,0,5,5])
            control_extractor = lambda x,u : u
            # constraints = [(scipy.optimize.LinearConstraint, constraint_A, constraint_lb, constraint_ub)]
            constraints = [(scipy.optimize.NonlinearConstraint, control_extractor, u_limits[0], u_limits[1])] * len(t)
        # /if
        traj_func = flatsys.point_to_point(self, t, x0=x0, xf=xf, constraints=constraints)
        s, u = traj_func.eval(t)
        return s, u
    # /plan()

# /class DifferentialDriveRobot2FlatSystem

class DifferentialDriveRobot2(Robot):
    
    def __init__(self, radius, wheel_radius, wheel_thickness):
        super(DifferentialDriveRobot2, self).__init__()
        self.R = radius
        self.r = wheel_radius
        self.wheel_thickness = wheel_thickness
        self.flatsys = DifferentialDriveRobot2FlatSystem(self.r, 2*self.R)
    # /__init__()

    def reset(self, x, y, θ_deg, ω_l, ω_r):
        self.s = np.array([[x, y, np.deg2rad(θ_deg), np.deg2rad(ω_l), np.deg2rad(ω_r)]])
    # /

    def stateDim(self):
        return 5
    #/

    def stateNames(self):
        return 'x', 'y', 'θ', 'ω_l', 'ω_r'
    #/

    def controlDim(self):
        return 2
    #/

    def controlNames(self):
        return 'α_l', 'α_r'
    #/

    def controlLimits(self):
        u_max = np.array([np.pi/18, np.pi/18]) # 10 deg/s^2
        u_min = -u_max
        return (u_min, u_max)
    # /controlLimits()

    @staticmethod
    def equationOfMotion(t, s, u, r, L):
        _, _, θ, ω_l, ω_r = s
        α_l, α_r = u
        v = (r/2) * (ω_r + ω_l)
        x_dot = v * np.cos(θ)
        y_dot = v * np.sin(θ)
        θ_dot = (r/L) * (ω_r - ω_l)
        ω_l_dot = α_l
        ω_r_dot = α_r
        return np.array([x_dot, y_dot, θ_dot, ω_l_dot, ω_r_dot])
    # /equationOfMotion()

    def parameters(self):
        return self.r, 2*self.R
    #/

    def dynamicsJacobian_state(self, s, u):
        J_s = np.zeros((self.stateDim(), self.stateDim()))
        _, _, θ, ω_l, ω_r = s
        α_l, α_r = u
        r = self.r
        half_r = r / 2
        L = 2 * self.R
        v = half_r * np.sum(ω_r + ω_l)
        sin_θ, cos_θ = np.sin(θ), np.cos(θ)
        J_s[0,2] = -v * sin_θ
        J_s[0,3] = half_r * cos_θ
        J_s[0,4] = half_r * cos_θ
        J_s[1,2] =  v * cos_θ
        J_s[1,3] = half_r * sin_θ
        J_s[1,4] = half_r * sin_θ
        J_s[2,3:5] = -(r/L), (r/L)
        return J_s
    # /dynamicsJacobian_state()

    def dynamicsJacobian_control(self, s, u):
        J_u = np.zeros((self.stateDim(), self.controlDim()))
        J_u[3:,:] = np.eye(2)
        return J_u
    # /dynamicsJacobian_control()

    def goto(self, s_goal, duration):
        self.gotoUsingIlqr(s_goal, duration, dt=0.01)
    # /goto()

    def gotoUsingIlqr(self, s_goal, duration, dt=0.01):
        self.fsmTransition(FsmState.PLANNING)
        N = int(duration / dt)
        t_dummy = np.nan # not used
        f = self.transitionFunction(dt)
        f_s = lambda s,u: np.eye(len(s)) + dt * self.dynamicsJacobian_state(s,u)
        f_u = lambda s,u: dt * self.dynamicsJacobian_control(s,u)
        P_N = 100 * np.eye(self.stateDim())
        Q = np.array([np.diag([1,1,1,1,1])] * N)
        # Q = np.eye(3) + (np.arange(N)/N)[:, np.newaxis, np.newaxis] * 0.01*P_N[np.newaxis, :, :]
        R_k = np.eye(self.controlDim())
        R_delta_u = 100 * np.eye(self.controlDim())
        s, u = iLQR(f, f_s, f_u,
                    self.s[-1], s_goal, N,
                    P_N, Q, R_k, R_delta_u)
        t = np.linspace(0,N,N+1) * dt
        self.setTrajectory(t,s,u)
    # /gotoUsingIlqr()

    def gotoUsingFlatsysToolbox(self, target_state, duration):
        self.fsmTransition(FsmState.PLANNING)
        N = int(duration / 0.01)
        timepts = np.linspace(0, duration, N)
        constraint_A = np.diag([0,0,0,1,1])
        constraint_lb = np.array([-5,-5]) # np.array([0,0,0,-5,-5])
        constraint_ub = np.array([ 5, 5]) # np.array([0,0,0,5,5])
        control_extractor = lambda x,u : u
        # constraints = [(scipy.optimize.LinearConstraint, constraint_A, constraint_lb, constraint_ub)]
        constraints = [(scipy.optimize.NonlinearConstraint, control_extractor, constraint_lb, constraint_ub)]
        cost = lambda x, u : np.dot(x - target_state, x - target_state) + np.dot(u,u)
        basis = flatsys.PolyFamily(8)
        traj_func = flatsys.point_to_point(self.flatsys, self.timepts[-1], x0 = self.s, xf=target_state, constraints=constraints, basis=basis, cost=cost)
        self.s, _ = traj_func.eval(timepts)
        # print('s.shape =', self.s.shape)
        self.setTrajectory(self.s.T)
    #/gotoUsingFlatsysToolbox()

    def renderCanonical(self, qpainter):
        '''
            Renders the robot in canonical coordinate frame.
            Call after setting the world transformation.
        '''
        brush = QtGui.QBrush()
        brush.setStyle(QtCore.Qt.SolidPattern)

        # main body
        brush.setColor(QtCore.Qt.black)
        qpainter.setBrush(brush)
        qpainter.drawEllipse(QtCore.QPoint(0, 0), self.R, self.R)
        
        # ... single dot to mark orientation
        brush.setColor(QtCore.Qt.yellow)
        qpainter.setBrush(brush)
        qpainter.drawEllipse(QtCore.QPoint(0.75 * self.R, 0), 0.1 * self.R, 0.1 * self.R)
        
        # wheels
        brush.setColor(QtCore.Qt.red)
        qpainter.setBrush(brush)
        qpainter.drawRect(-self.R, -self.R -self.wheel_thickness, 2 * self.R, self.wheel_thickness)
        qpainter.drawRect(-self.R,  self.R,                       2 * self.R, self.wheel_thickness)

    # /renderCanonical()

# /class DifferentialDriveRobot2
