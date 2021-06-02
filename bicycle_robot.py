import numpy as np
import scipy.integrate
import time
import copy
from PyQt5 import QtGui, QtCore

from fsm_state import FsmState

class BicycleRobot:
    def __init__(self, wheel_radius, baseline) -> None:
        self.r = wheel_radius
        self.L = baseline
        self.fsm_state = FsmState.IDLE
        self.s_counter = 0
        pass
    # /__init_()

    def reset(self, x, y, θ_deg, φ_deg):
        self.s = np.array([[x, y, np.deg2rad(θ_deg), np.deg2rad(φ_deg)]])
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
        u_max = np.array([30.0, np.deg2rad(60.0)]) # pix/s, 60 deg/s
        u_min = -u_max
        return u_min, u_max
    # /controlLimits()

    def fsmTransition(self, fsm_state):
        print('State transition: ', self.fsm_state, '->', fsm_state)
        self.fsm_state = fsm_state
    # fsmTransition()

    @staticmethod
    def equationOfMotion(t, s, u, params):
        _, _, θ, φ = s
        v, φ_dot = u
        r = params['r'] # wheel radius
        L = params['L'] # baseline
        x_dot = v * np.cos(θ)
        y_dot = v * np.sin(θ)
        θ_dot = (v/L) * np.tan(φ)
        φ_dot = φ_dot
        return np.array([x_dot, y_dot, θ_dot, φ_dot])
    # /equationOfMotion()

    def transitionFunction(self, dt):
        return lambda s,u: s + dt * self.equationOfMotion(np.nan, s, u,
                                                          params={'r': self.r,
                                                                  'L': self.L})
    # /transitionFunction()

    def dynamics(self, t, s, u):
        # SciPy's odeint() needs 't' in signature
        return self.equationOfMotion(t, s, u,
                                     params={'r': self.r,
                                             'L': self.L})
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

        x, y, θ, _ = self.currentPose()
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
        brush = QtGui.QBrush()
        brush.setStyle(QtCore.Qt.SolidPattern)

        # rear wheel
        brush.setColor(QtCore.Qt.red)
        qpainter.setBrush(brush)
        qpainter.drawRect(-self.r, -5, 2*self.r, 10)

        # front wheel
        original_xform = qpainter.worldTransform()
        front_wheel_xform = copy.deepcopy(original_xform)
        front_wheel_xform.translate(self.L, 0)
        φ = self.s[self.s_counter,3]
        front_wheel_xform.rotate(np.rad2deg(-φ))
        qpainter.setWorldTransform(front_wheel_xform)
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
