import time
import numpy as np
import scipy
from PyQt5 import QtGui, QtCore

from fsm_state import FsmState

class Ellipse:
    def __init__(self, a, b, φ0_deg):
        self.a = a # major axis
        self.b = b # minor axis
        self.φ0 = φ0_deg * np.pi / 180.0 # phase/angular orientation w.r.t. x-axis
    # __init__()

    def r(self, φ):
        x = self.a * np.cos(self.φ0 + φ) 
        y = self.b * np.sin(self.φ0 + φ)
        r = np.sqrt(x*x + y*y)
        return r
    # __call__()
# /Ellipse

class DifferentialDriveEllipseWheelRobot:
    
    def __init__(self, baseline, left_wheel_ellipse, right_wheel_ellipse, wheel_thickness, initial_xyθ = (0,0,0)):
        self.fsm_state = FsmState.IDLE
        self.L = baseline
        self.l_wheel = left_wheel_ellipse
        self.r_wheel = right_wheel_ellipse
        self.wheel_thickness = wheel_thickness
        self.plot_history = False

        x, y, θ = initial_xyθ
        self.reset(x, y, θ)
        # /if
        self.t = 0 # relative time within a driving mission

    # /__init__()

    def reset(self, x, y, θ):
        self.s = np.array([x, y, θ, 0, 0])
    # /

    def plotHistory(self, enable=True):
        self.plot_history = enable
    # /

    def fsmTransition(self, fsm_state):
        print('State transition: ', self.fsm_state, '->', fsm_state)
        self.fsm_state = fsm_state
    # /fsmTransition()

    def dynamics(self, t, s, u, params):
        _, _, θ, φ_l, φ_r = s
        ω_l, ω_r = u
        v_l = self.l_wheel.r(φ_l) * ω_l
        v_r = self.r_wheel.r(φ_r) * ω_r
        v = (v_l + v_r) / 2
        x_dot = v * np.cos(θ)
        y_dot = v * np.sin(θ)
        θ_dot = (v_r - v_l) / self.L
        φ_l_dot = ω_l
        φ_r_dot = ω_r
        return np.array([x_dot, y_dot, θ_dot, φ_l_dot, φ_r_dot])
    # /dynamics()

    def applyControl(self, delta_t, u):
        '''
            u: angular velocities of left and right wheels, respectively, in rad/s
        '''
        self.s = scipy.integrate.odeint(self.dynamics,
                                        self.s,
                                        np.array([0, delta_t]),
                                        args=(u,{}),
                                        tfirst=True)[1]
    # /applyControls()

    def go(self, u_s, t_s):
        if len(u_s) == len(t_s) and self.fsm_state == FsmState.IDLE:
            self.s_counter = 0
            self.t_drive_begin = time.time()
            self.u_s = u_s
            self.t_s = np.cumsum(t_s)
            self.t = 0.0
            self.s_history = []
            self.fsmTransition(FsmState.DRIVING)
        else:
            print('Cannot go: u_s and t_s of unequal lengths')
        # /if
    # /go()

    # Draw this instance onto a qpainter
    def render(self, qpainter, window_height):
        t_drive = time.time() - self.t_drive_begin
        # print('render @', t_drive)
        if t_drive > self.t_s[-1]:
            self.fsmTransition(FsmState.IDLE)

        else:
            while self.s_counter < len(self.t_s) and self.t_s[self.s_counter] < t_drive:
                delta_t = self.t_s[self.s_counter] - self.t
                self.applyControl(delta_t, self.u_s[self.s_counter])
                self.t = self.t_s[self.s_counter]
                self.s_counter += 1
            #/
            assert self.s_counter < len(self.t_s)
            delta_t = t_drive - self.t
            self.applyControl(delta_t, self.u_s[self.s_counter])
            self.t = t_drive
            self.s_history.append(self.s)
        # /if-else

        if self.plot_history:
            for s in self.s_history:
                qpainter.drawPoint(s[0], window_height -1 -s[1])
        x, y, θ, _, _ = self.s
        original_transform = qpainter.worldTransform()
        transform = QtGui.QTransform()
        transform.translate(x, window_height -1 -y)
        transform.rotate(-θ * 180 / np.pi) # degrees
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

# /class DifferentialDriveRobot
