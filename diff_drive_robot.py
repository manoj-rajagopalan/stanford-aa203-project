import time
from jax.dtypes import dtype

import numpy as np
import scipy.integrate
import scipy.optimize
import control.flatsys as flatsys

from PyQt5 import QtGui, QtCore

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
    # /flatsysForward()

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
    # /flatsysReverse()

# /class DifferentialDriveRobotFlatSystem

class DifferentialDriveRobot:
    
    def __init__(self, radius, wheel_radius, wheel_thickness):
        self.fsm_state = FsmState.IDLE
        self.radius = radius
        self.wheel_radius = wheel_radius
        self.wheel_thickness = wheel_thickness

        self.s_counter = 0

        self.flatsys = DifferentialDriveRobotFlatSystem(self.wheel_radius, 2*self.radius)

        # self.flatsys = flatsys.FlatSystem(self.flatsysForward,
        #                                   self.flatsysReverse,
        #                                   updfcn=self.dynamics,
        #                                   params={'r': self.wheel_radius, 'L': 2*self.radius},
        #                                   inputs=['omega_l', 'omega_r'],
        #                                   outputs=['flat_x', 'flat_y'],
        #                                   states=['x', 'y', 'theta'])
    # /__init__()

    def reset(self, x, y, θ_deg):
        self.s = np.array([[x, y, np.deg2rad(θ_deg)]])
    # /

    def stateDim(self):
        return 3
    #/

    def controlDim(self):
        return 2
    #/

    def controlLimits(self):
        u_max = np.array([30.0, 30.0]) # rad/s
        u_min = -u_max
        return u_min, u_max
    # /controlLimits()

    def fsmTransition(self, fsm_state):
        print('State transition: ', self.fsm_state, '->', fsm_state)
        self.fsm_state = fsm_state
    # fsmTransition()

    @staticmethod
    def equationOfMotion(t, s, u, params):
        _, _, θ = s
        ω_l, ω_r = u
        r = params['r'] # wheel radius
        L = params['L'] # baseline
        v = (r/2) * (ω_r + ω_l)
        x_dot = v * np.cos(θ)
        y_dot = v * np.sin(θ)
        θ_dot = (r/L) * (ω_r - ω_l)
        return np.array([x_dot, y_dot, θ_dot])
    # /equationOfMotion()

    def transitionFunction(self, dt):
        return lambda s,u: s + dt * self.equationOfMotion(np.nan, s, u,
                                                          params={'r': self.wheel_radius,
                                                                  'L': 2*self.radius})
    # /transitionFunction()

    def dynamics(self, t, s, u):
        # SciPy's odeint() needs 't' in signature
        return self.equationOfMotion(t, s, u,
                                     params={'r': self.wheel_radius,
                                             'L': 2*self.radius})
    # /dynamics

    def applyControl(self, delta_t, s, u):
        '''
            u: angular velocities of left and right wheels, respectively, in rad/s
        '''
        s = scipy.integrate.odeint(self.dynamics,
                                   s,
                                   np.array([0, delta_t]),
                                   args=(u,),
                                   tfirst=True)[1]
        return s
    # /applyControls()

    def dynamicsJacobian_state(self, s, u):
        J_s = np.zeros((3,3))
        θ = s[2]
        r = self.wheel_radius
        v = r/2 * np.sum(u)
        J_s[0,2] = -v * np.sin(θ)
        J_s[1,2] =  v * np.cos(θ)
        return J_s
    # /dynamicsJacobian_state()

    def dynamicsJacobian_control(self, s, u):
        J_u = np.zeros((3,2))
        θ = s[2]
        r = self.wheel_radius
        L = 2 * self.radius # baseline
        J_u[0,:].fill(r/2 * np.cos(θ))
        J_u[1,:].fill(r/2 * np.sin(θ))
        J_u[2,0] = -r/L
        J_u[2,1] = r/L
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
        P_N = 5000 * np.eye(3)
        # Q_k = np.diag([1,1,1])
        Q = np.eye(3) + (np.arange(N)/N)[:, np.newaxis, np.newaxis] * 0.01*P_N[np.newaxis, :, :]
        R_k = 5 * np.eye(2)
        R_delta_u = 100 * np.eye(2)
        s, _ = iLQR(f, f_s, f_u,
                    self.s[-1], s_goal, N,
                    P_N, Q, R_k, R_delta_u)
        t = np.linspace(0,N,N+1) * dt
        self.setTrajectory(s,t)
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
    # /gotoUsingFlatsysToolbox()

    def setTrajectory(self, s,t):
        self.s = s
        self.timepts = t
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

    # Draw this instance onto a qpainter
    def render(self, qpainter, window_height):
        if self.fsm_state == FsmState.PLANNING:
            return
        # /if


        if self.fsm_state == FsmState.DRIVING:
            t_drive = time.time() - self.t_drive_begin
            while self.s_counter < len(self.s) and self.timepts[self.s_counter] < t_drive:
                self.s_counter += 1
            #/
            if self.s_counter == len(self.timepts):
                self.fsmTransition(FsmState.IDLE)
            # /if

            self.s_counter -= 1
        # /if

        x, y, θ = self.currentPose()
        original_transform = qpainter.worldTransform()
        transform = QtGui.QTransform()
        transform.translate(x, window_height -1 -y)
        transform.rotate(np.rad2deg(-θ)) # degrees
        qpainter.setWorldTransform(transform, combine=False)

        self.renderCanonical(qpainter, window_height)

        qpainter.setWorldTransform(original_transform)
        
    # /render()

    def renderCanonical(self, qpainter, window_height):
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
        qpainter.drawRect(-self.wheel_radius, -self.radius -self.wheel_thickness, 2 * self.wheel_radius, self.wheel_thickness)
        qpainter.drawRect(-self.wheel_radius,  self.radius,                       2 * self.wheel_radius, self.wheel_thickness)

        # single dot to mark orientation
        brush.setColor(QtCore.Qt.yellow)
        qpainter.setBrush(brush)
        qpainter.drawEllipse(QtCore.QPoint(0.75 * self.radius, 0), 0.1 * self.radius, 0.1 * self.radius)
    # /renderCanonical()

# /class DifferentialDriveRobot
