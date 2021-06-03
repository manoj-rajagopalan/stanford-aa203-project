import numpy as np
import scipy.integrate
import time
import copy
from PyQt5 import QtGui, QtCore

import control.flatsys as flatsys

from fsm_state import FsmState
from ilqr import iLQR

class BicycleRobot2FlatSystem(flatsys.FlatSystem):
    def __init__(self, r, L):
        self.r = r # wheel radius
        self.L = L # baseline
        super(BicycleRobot2FlatSystem, self).__init__(self.forward,
                                                     self.reverse,
                                                     inputs=['v', 'φ_dot'],
                                                     outputs=['flat_x', 'flat_y'],
                                                     states=['x', 'y', 'θ', 'φ'])
    # /__init__()

    def forward(self, s, u):
        r, L = self.r, self.L
        x, y, θ, φ = s
        v, φ_dot = u
        x_dot = v * np.cos(θ)
        y_dot = v * np.sin(θ)
        θ_dot = (v / L) * np.tan(φ)
        x_ddot = -y_dot * θ_dot
        y_ddot = x_dot * θ_dot
        cos_φ = np.cos(φ)
        sec2_φ = 1 / (cos_φ * cos_φ)
        θ_ddot = (v/L) * sec2_φ * φ_dot
        x_dddot = -y_ddot * θ_dot - y_dot * θ_ddot
        y_dddot = x_ddot * θ_dot + x_dot * θ_ddot
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
        φ = np.arctan2(L * (x_dot*y_ddot - x_ddot*y_ddot), v * v_sqr)
        tan_φ = np.tan(φ) # for ease in next calculations
        φ_dot = 1/(1 + tan_φ*tan_φ) * (L/v) * ((x_dot * y_dddot - x_dddot * y_dddot) - tan_φ*2*(x_dot*x_ddot + y_dot*y_ddot)) / v_sqr
        s = np.array([x, y, θ, φ])
        u = np.array([v, φ_dot])
        return s, u
    # /reverse()

    def plan(self, s0, sf, timepts):
        x0 = copy.deepcopy(s0)
        x0[2] = np.deg2rad(s0[2])
        xf = copy.deepcopy(sf)
        xf[2] = np.deg2rad(sf[2])
        traj_func = flatsys.point_to_point(self, timepts, x0=x0, xf=xf)
        s, u = traj_func.eval(timepts)
        return s, u
    # /plan()

# /class BicycleRobot2FlatSystem


class BicycleRobot2:
    def __init__(self, wheel_radius, baseline) -> None:
        self.r = wheel_radius
        self.L = baseline
        self.fsm_state = FsmState.IDLE
        self.s_counter = 0
    # /__init_()

    def reset(self, x, y, θ_deg, φ_deg):
        self.s = np.array([[x, y, np.deg2rad(θ_deg), np.deg2rad(φ_deg)]])
        self.u = np.zeros((1,self.controlDim()))
        self.timepts = np.array([0])
        self.fsm_state = FsmState.IDLE
    # /

    def stateDim(self):
        return 4
    #/

    def controlDim(self):
        return 2
    #/

    def controlLimits(self):
        u_max = np.array([30.0, np.deg2rad(15.0)]) # pix/s, 15 deg/s
        u_min = -u_max
        return u_min, u_max
    # /controlLimits()

    def fsmTransition(self, fsm_state):
        print('State transition: ', self.fsm_state, '->', fsm_state)
        self.fsm_state = fsm_state
    # fsmTransition()

    @staticmethod
    def equationOfMotion(t, s, u, r, L):
        _, _, θ, φ = s
        v, φ_dot = u
        x_dot = v * np.cos(θ)
        y_dot = v * np.sin(θ)
        θ_dot = (v/L) * np.tan(φ)
        φ_dot = φ_dot
        return np.array([x_dot, y_dot, θ_dot, φ_dot])
    # /equationOfMotion()

    def transitionFunction(self, dt):
        return lambda s,u: s + dt * self.equationOfMotion(np.nan, s, u, self.r, self.L)
    # /transitionFunction()

    # def dynamics(self, t, s, u):
    #     # SciPy's odeint() needs 't' in signature
    #     return self.equationOfMotion(t, s, u,
    #                                  params={'r': self.r,
    #                                          'L': self.L})
    # # /dynamics

    def applyControl(self, delta_t, s, u):
        '''
            u: angular velocities of left and right wheels, respectively, in rad/s
        '''
        s = scipy.integrate.odeint(self.equationOfMotion,
                                   s,
                                   np.array([0, delta_t]),
                                   args=(u, self.r, self.L),
                                   tfirst=True)[1]
        return s
    # /applyControls()

    def dynamicsJacobian_state(self, s, u):
        x, y, θ, φ = s
        v, _ = u
        J_s = np.zeros((self.stateDim(), self.stateDim()))
        J_s[0,2] = -y
        J_s[1,2] =  x
        cos_φ = np.cos(φ)
        sec2_φ = 1 / (cos_φ * cos_φ)
        J_s[2,3] = (v/self.L) * sec2_φ
        return J_s
    # /dynamicsJacobian_state()

    def dynamicsJacobian_control(self, s, u):
        _, _, θ, φ = s
        v, _ = u
        J_u = np.zeros((self.stateDim(), self.controlDim()))
        J_u[0,0] = np.cos(θ)
        J_u[1,0] = np.sin(θ)
        tan_φ = np.tan(φ)
        J_u[2,0] = (1/self.L) * tan_φ
        J_u[2,1] = 1
        return J_u
    # /dynamicsJacobian_control()

    def setTrajectory(self, s,u,t):
        N = len(t)
        assert N == len(s)
        assert (N-1) == len(u)
        self.fsmTransition(FsmState.IDLE)
        self.s = s
        self.u = u
        self.timepts = t
        for i in range(N-1):
            self.s[i+1] = scipy.integrate.odeint(self.equationOfMotion,
                                                 self.s[i],
                                                 np.array([t[i], t[i+1]]),
                                                 args=(u[i], self.r, self.L),
                                                 tfirst=True)[1]
        # /for i
        self.t = t[1:]
        self.drive()
    # /setTrajectory()

    def drive(self):
        self.t_drive_begin = time.time()
        self.s_counter = 0
        self.fsmTransition(FsmState.DRIVING)
    # /drive()

    def currentPose(self):
        return self.s[self.s_counter]
    # /

    def goto(self, s_goal, duration):
        self.gotoUsingIlqr(s_goal, duration, dt=0.01)
    # /goto()

    def gotoUsingIlqr(self, s_goal, duration, dt=0.01):
        self.fsmTransition(FsmState.PLANNING)
        s_goal[2:] = np.deg2rad(s_goal[2:])
        N = int(duration / dt)
        t_dummy = np.nan # not used
        f = self.transitionFunction(dt)
        f_s = lambda s,u: np.eye(len(s)) + dt * self.dynamicsJacobian_state(s,u)
        f_u = lambda s,u: dt * self.dynamicsJacobian_control(s,u)
        P_N = 50 * np.eye(self.stateDim())
        Q = np.array([np.diag([1,1,1,1])] * N)
        # Q = np.eye(3) + (np.arange(N)/N)[:, np.newaxis, np.newaxis] * 0.01*P_N[np.newaxis, :, :]
        R_k = 0.1 * np.eye(self.controlDim())
        R_delta_u = 1000 * np.eye(self.controlDim())
        s, u = iLQR(f, f_s, f_u,
                    self.s[-1], s_goal, N,
                    P_N, Q, R_k, R_delta_u)
        t = np.linspace(0,N,N+1) * dt
        self.setTrajectory(s,u,t)
    # /gotoUsingIlqr()

    def render(self, qpainter, window_height):
        if self.fsm_state == FsmState.PLANNING:
            return
        # /if

        if self.fsm_state == FsmState.DRIVING:
            t_drive = time.time() - self.t_drive_begin
            while self.s_counter < len(self.timepts) and self.timepts[self.s_counter] < t_drive:
                self.s_counter += 1
            #/
            if self.s_counter == len(self.timepts):
                self.fsmTransition(FsmState.IDLE)
            #/
        # /if

        original_transform = qpainter.worldTransform()
        # transform = QtGui.QTransform()
        # transform.translate(x, window_height -1 -y)
        # transform.rotate(np.rad2deg(-θ)) # degrees
        # qpainter.setWorldTransform(transform, combine=False)

        # to vehicle pose
        x, y, θ, _= self.s[self.s_counter]
        qpainter.translate(x,y)
        qpainter.rotate(np.rad2deg(θ)) # qpainter rotates clockwise and in degrees
        self.renderCanonical(qpainter, window_height)

        qpainter.setWorldTransform(original_transform)
        
    # /render()

    def renderCanonical(self, qpainter, window_height):
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
        φ = self.s[self.s_counter, 3]
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
