import time
import numpy as np
import scipy.integrate
from PyQt5 import QtGui, QtCore

from fsm_state import FsmState
from models.model import Model
class Robot:
    def __init__(self, model) -> None:
        self.model = model
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

    def stateDim(self):
        return self.model.stateDim()
    #/

    def stateNames(self):
        return self.model.stateNames()
    #/

    def controlDim(self):
        return self.model.controlDim()
    #/

    def controlNames(self):
        return self.model.controlNames()
    #/

    def parameters(self):
        return self.model.parameters()
    #/

    def dynamics(self, t, s, u):
        return self.model.dynamics(t,s,u)
    #/

    def controlLimits(self): # override
        raise NotImplementedError
    #/

    def transitionFunction(self, dt):
        return lambda s,u: s + dt * self.model.dynamics(np.nan, s, u)
    # /transitionFunction()

    def applyControl(self, delta_t, s, u):
        '''
            u: angular velocities of left and right wheels, respectively, in rad/s
        '''
        s = scipy.integrate.odeint(self.model.dynamics,
                                   s,
                                   np.array([0, delta_t]),
                                   args=(u,),
                                   tfirst=True)[1]
        return s
    # /applyControl()

    def dynamicsJacobianWrtState(self, s, u):
        return self.model.dynamicsJacobianWrtState(s,u)
    #/

    def dynamicsJacobianWrtControl(self, s, u):
        self.model.dynamicsJacobianWrtControl(s,u)
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

