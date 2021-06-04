import time
import numpy as np
import scipy.integrate
from PyQt5 import QtGui, QtCore

from fsm_state import FsmState

class Robot:
    def __init__(self) -> None:
        self.fsm_state = FsmState.IDLE

        # Trajectory
        self.s_counter = 0
        self.s = None
        self.u = None
        self.t = None
    #/__init__()
    
    def fsmTransition(self, fsm_state):
        print('State transition: ', self.fsm_state, '->', fsm_state)
        self.fsm_state = fsm_state
    #/fsmTransition()

    def reset(self, *args):
        raise NotImplementedError
    #/

    def stateDim(self): # override
        raise NotImplementedError
    #/

    def stateNames(self): # override
        raise NotImplementedError
    #/

    def controlDim(self): # override
        raise NotImplementedError
    #/

    def controlNames(self): # oveerride
        raise NotImplementedError
    #/

    def controlLimits(self): # override
        raise NotImplementedError
    #/

    def parameters(self): # override
        raise NotImplementedError
    #/

    @staticmethod
    def equationOfMotion(t, s, u, *args): # override
        raise NotImplementedError
    #/

    def transitionFunction(self, dt):
        return lambda s,u: s + dt * self.equationOfMotion(np.nan, s, u, *self.parameters())
    # /transitionFunction()

    def applyControl(self, delta_t, s, u):
        '''
            u: angular velocities of left and right wheels, respectively, in rad/s
        '''
        args = tuple([u]) + self.parameters()
        # ... u is of type ndarray
        #     we want to wrap it in a tuple to be able to add parameters to that tuple
        #     tuple(u) will convert the ndarray into a tuple instead of wrapping it
        #     hence wrap in list first and then convert.

        s = scipy.integrate.odeint(self.equationOfMotion,
                                   s,
                                   np.array([0, delta_t]),
                                   args=args,
                                   tfirst=True)[1]
        return s
    # /applyControl()

    def dynamicsJacobian_state(self, s, u): # override
        raise NotImplementedError
    #/

    def dynamicsJacobian_control(self, s, u): # overridee
        raise NotImplementedError
    #/

    def gotoUsingIlqr(self, s_goal, duration, dt): # override
        raise NotImplementedError
    #/

    def setTrajectory(self, t,s,u):
        N = len(t)
        assert t.shape == (N,)
        assert N == len(s)
        assert (N-1) == len(u)
        self.t = t
        self.s = s
        self.u = u
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

    def renderCanonical(self, qpainter): # override
        raise NotImplementedError
    #/

    def render(self, qpainter):
        if self.fsm_state == FsmState.DRIVING:
            t_drive = time.time() - self.t_drive_begin
            while self.s_counter < len(self.t) and self.t[self.s_counter] < t_drive:
                self.s_counter += 1
            #/
            if self.s_counter == len(self.t):
                self.fsmTransition(FsmState.IDLE)
            #/
            self.s_counter -= 1
        # /if

        original_transform = qpainter.worldTransform()

        # to vehicle pose
        x, y, θ = self.s[self.s_counter, 0:3]
        qpainter.translate(x,y)
        qpainter.rotate(np.rad2deg(θ)) # qpainter rotates clockwise and in degrees
        self.renderCanonical(qpainter)

        qpainter.setWorldTransform(original_transform)
        
    # /render()

    def plotTrajectory(self, state_plot, control_plot): # override
        raise NotImplementedError
    #/
#/class Robot

