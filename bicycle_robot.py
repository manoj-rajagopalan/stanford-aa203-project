import time
import copy

import numpy as np
import scipy.integrate
from PyQt5 import QtGui, QtCore
import control.flatsys as flatsys

from robot import Robot
from fsm_state import FsmState
from ilqr import iLQR
class BicycleRobotFlatSystem(flatsys.FlatSystem):
    def __init__(self, r, L):
        self.r = r # wheel radius
        self.L = L # baseline
        super(BicycleRobotFlatSystem, self).__init__(self.forward,
                                                     self.reverse,
                                                     inputs=['v', 'φ'],
                                                     outputs=['flat_x', 'flat_y'],
                                                     states=['x', 'y', 'θ'])
    # /__init__()

    def forward(self, s, u):
        r, L = self.r, self.L
        x, y, θ = s
        v, φ = u
        x_dot = v * np.cos(θ)
        y_dot = v * np.sin(θ)
        θ_dot = (v / L) * np.tan(φ)
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
        φ = np.arctan2(L * (x_dot*y_ddot - x_ddot*y_ddot), v * v_sqr)
        s = np.array([x, y, θ])
        u = np.array([v, φ])
        return s, u
    # /reverse()

    def plan(self, s0, sf, timepts):
        x0 = copy.deepcopy(s0)
        x0[2] = np.deg2rad(s0[2])
        xf = copy.deepcopy(sf)
        xf[2] = np.deg2rad(sf[2])
        traj_func = flatsys.point_to_point(self, timepts, x0=x0, xf=xf)
        s, u = traj_func.eval(timepts)
        return s.T, u.T[:-1]
    # /plan()

# /class BicycleRobotFlatSystem

class BicycleRobot(Robot):
    def __init__(self, wheel_radius, baseline) -> None:
        super(BicycleRobot, self).__init__()
        self.r = wheel_radius
        self.L = baseline
    # /__init_()

    def reset(self, x, y, θ_deg):
        self.s = np.array([[x, y, np.deg2rad(θ_deg)]])
        self.u = np.zeros((1,self.controlDim()))
        self.timepts = np.array([0])
        self.fsm_state = FsmState.IDLE
    # /

    def stateDim(self):
        return 3
    #/

    def stateNames(self):
        return 'x', 'y', 'θ'
    #/

    def controlDim(self):
        return 2
    #/

    def controlNames(self):
        return 'v', 'φ'
    #/

    def controlLimits(self):
        u_max = np.array([50.0, np.deg2rad(60.0)]) # pix/s, 30 deg
        u_min = -u_max
        return u_min, u_max
    # /controlLimits()

    def parameters(self):
        return (self.r, self.L)
    #/

    @staticmethod
    def equationOfMotion(t, s, u, r, L):
        _, _, θ = s
        v, φ = u
        x_dot = v * np.cos(θ)
        y_dot = v * np.sin(θ)
        θ_dot = (v/L) * np.tan(φ)
        return np.array([x_dot, y_dot, θ_dot])
    # /equationOfMotion()

    def dynamicsJacobian_state(self, s, u):
        x, y, θ = s
        x_dot, y_dot, _ = self.equationOfMotion(np.nan, s, u, *self.parameters())
        J_s = np.zeros((self.stateDim(), self.stateDim()))
        J_s[0,2] = -y_dot
        J_s[1,2] =  x_dot
        return J_s
    # /dynamicsJacobian_state()

    def dynamicsJacobian_control(self, s, u):
        _, _, θ = s
        v, φ = u
        J_u = np.zeros((self.stateDim(), self.controlDim()))
        J_u[0,0] = np.cos(θ)
        J_u[1,0] = np.sin(θ)
        tan_φ = np.tan(φ)
        J_u[2,0] = (1/self.L) * tan_φ
        J_u[2,1] = (v/self.L) * (1 + tan_φ * tan_φ)
        return J_u
    # /dynamicsJacobian_control()

    def gotoUsingIlqr(self, s_goal, duration, dt=0.01):
        self.fsmTransition(FsmState.PLANNING)
        s_goal[2] = np.deg2rad(s_goal[2])
        N = int(duration / dt)
        t_dummy = np.nan # not used
        f = self.transitionFunction(dt)
        f_s = lambda s,u: np.eye(len(s)) + dt * self.dynamicsJacobian_state(s,u)
        f_u = lambda s,u: dt * self.dynamicsJacobian_control(s,u)
        P_N = 50 * np.eye(self.stateDim())
        Q = np.array([np.diag([1,1,1])] * N)
        # Q = np.eye(3) + (np.arange(N)/N)[:, np.newaxis, np.newaxis] * 0.01*P_N[np.newaxis, :, :]
        R_k = 0.1 * np.eye(self.controlDim())
        R_delta_u = 5000 * np.eye(self.controlDim())
        s, u = iLQR(f, f_s, f_u,
                    self.s[-1], s_goal, N,
                    P_N, Q, R_k, R_delta_u)
        t = np.linspace(0,N,N+1) * dt
        self.setTrajectory(t,s,u)
    # /gotoUsingIlqr()

    def renderCanonical(self, qpainter):
        '''
            Renders the robot in canonical coordinate frame.
            Call after setting the world transformation.
        '''
        brush = QtGui.QBrush()
        brush.setStyle(QtCore.Qt.SolidPattern)

        # rear wheel
        brush.setColor(QtCore.Qt.red)
        qpainter.setBrush(brush)
        qpainter.drawRect(-self.r, -5, 2*self.r, 10)

        # front wheel
        original_xform = qpainter.worldTransform()
        qpainter.translate(self.L, 0)
        φ = self.u[min(self.s_counter, len(self.u)-1), 1]
        qpainter.rotate(np.rad2deg(φ)) # qpainter rotates clockwise and in degrees
        qpainter.drawRect(-self.r, -5, 2*self.r, 10)
        qpainter.setWorldTransform(original_xform)

        # main body
        brush.setColor(QtCore.Qt.black)
        qpainter.setBrush(brush)
        qpainter.drawRect(0, -10, self.L, 20)

        # single dot to mark orientation
        brush.setColor(QtCore.Qt.yellow)
        qpainter.setBrush(brush)
        qpainter.drawEllipse(QtCore.QPoint(0.75 * self.L, 0), 10, 10)
    # /renderCanonical()

# /class BicycleRobot
