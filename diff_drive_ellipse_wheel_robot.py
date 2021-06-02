import time
import numpy as np
import scipy
from PyQt5 import QtGui, QtCore

from fsm_state import FsmState

class Ellipse:
    def __init__(self, a, b, φ0_deg):
        self.a = a # major axis
        self.b = b # minor axis
    # __init__()

    def r(self, φ):
        x = self.a * np.cos(φ)
        y = self.b * np.sin(φ)
        r = np.sqrt(x*x + y*y)
        return r
    # __call__()
# /Ellipse

class DifferentialDriveEllipseWheelRobot:
    
    def __init__(self, baseline, left_wheel_ellipse, right_wheel_ellipse, wheel_thickness):
        self.fsm_state = FsmState.IDLE
        self.L = baseline
        self.l_wheel = left_wheel_ellipse
        self.r_wheel = right_wheel_ellipse
        self.wheel_thickness = wheel_thickness
        self.plot_history = False
        self.t = 0 # relative time within a driving mission

    # /__init__()

    def reset(self, x, y, θ_deg, φ_l_deg, φ_r_deg):
        self.s = np.array([[x, y, np.deg2rad(θ_deg), np.deg2rad(φ_l_deg) , np.deg2rad(φ_r_deg)]])
    # /

    def stateDim(self):
        return 5
    #/

    def controlDim(self):
        return 2
    #/

    def controlLimits(self):
        u_max = np.array([30.0, 30.0]) # rad/s
        u_min = -u_max
        return u_min, u_max
    # /controlLimits()

    def plotHistory(self, enable=True):
        self.plot_history = enable
    # /

    def fsmTransition(self, fsm_state):
        if fsm_state != self.fsm_state:
            print('State transition: ', self.fsm_state, '->', fsm_state)
            self.fsm_state = fsm_state
        # /if
    # /fsmTransition()

    @staticmethod
    def equationOfMotion(self, t, s, u, params):
        _, _, θ, φ_l, φ_r = s
        ω_l, ω_r = u
        l_wheel = params['l_wheel']
        r_wheel = params['r_wheel']
        L = params['L']
        v_l = l_wheel.r(φ_l) * ω_l
        v_r = r_wheel.r(φ_r) * ω_r
        v = (v_l + v_r) / 2
        x_dot = v * np.cos(θ)
        y_dot = v * np.sin(θ)
        θ_dot = (v_r - v_l) / L
        φ_l_dot = ω_l
        φ_r_dot = ω_r
        return np.array([x_dot, y_dot, θ_dot, φ_l_dot, φ_r_dot])
    # /equationOfMotion()

    def transitionFunction(self, dt):
        return lambda s,u: s + dt * self.equationOfMotion(np.nan, s, u,
                                                          params={'l_wheel': self.l_wheel,
                                                                  'r_wheel': self.r_wheel,
                                                                  'L': 2*self.radius})
    # /transitionFunction()

    def dynamics(self, t, s, u):
        # SciPy's odeint() needs 't' in signature
        return self.equationOfMotion(t, s, u,
                                     params={'l_wheel': self.l_wheel,
                                             'r_wheel': self.r_wheel,
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

    def go(self, u_s, t_s):
        N = len(t_s)
        if len(u_s) == N and self.fsm_state == FsmState.IDLE:
            s = np.empty((N+1, 5))
            s[0] = self.s[-1]
            t_prev = 0
            for n in range(N):
                delta_t = t_s[n] - t_prev
                s[n+1] = self.applyControl(delta_t, self.s[n], u_s[n])
                t_prev = t_s[n]
            # /for n
            self.setTrajectory(s, t_s)
        else:
            print('Cannot go: u_s and t_s of unequal lengths')
        # /if-else
    # /go()

    def setTrajectory(self, s,t):
        self.fsmTransition(FsmState.IDLE)
        self.s = s
        self.timepts = t
        self.drive()
    # /setTrajectory()

    def drive(self):
        self.t_drive_begin = time.time()
        self.s_counter = 0
        self.s_history = []
        self.fsmTransition(FsmState.DRIVING)
    # /drive()

    def currentPose(self):
        return self.s[self.s_counter, 0:3]
    # /

    # Draw this instance onto a qpainter
    def render(self, qpainter, window_height):
        t_drive = time.time() - self.t_drive_begin

        if self.plot_history:
            for s in self.s_history:
                qpainter.drawPoint(s[0], window_height -1 -s[1])
            # /for
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

        x, y, θ, = self.currentPose()
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

        # main body
        qpainter.drawEllipse(QtCore.QPoint(0, 0), self.L/2, self.L/2)
        # - single dot to mark orientation
        brush.setColor(QtCore.Qt.yellow)
        qpainter.setBrush(brush)
        qpainter.drawEllipse(QtCore.QPoint(0.75 * self.L/2, 0), 0.1 * self.L/2, 0.1 * self.L/2)

        # wheels
        φ_l = self.s[3]
        l_wheel_halflen = np.max(np.abs(np.array([self.l_wheel.a * np.cos(self.l_wheel.φ0 + φ_l),
                                                  self.l_wheel.b * np.sin(self.l_wheel.φ0 + φ_l)])))
        qpainter.drawRect(-l_wheel_halflen, -self.L/2 -self.wheel_thickness, 2 * l_wheel_halflen, self.wheel_thickness)

        φ_r = self.s[4]
        r_wheel_halflen = np.max(np.abs(np.array([self.r_wheel.a * np.cos(self.r_wheel.φ0 + φ_r),
                                                  self.r_wheel.b * np.sin(self.r_wheel.φ0 + φ_r)])))
        qpainter.drawRect(-r_wheel_halflen, self.L/2, 2 * r_wheel_halflen, self.wheel_thickness)
    # /renderCanonical()

# /class DifferentialDriveEllipseWheelRobot
