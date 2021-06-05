import time
import bisect

import numpy as np
import scipy.integrate
from PyQt5 import QtGui, QtCore

from fsm_state import FsmState
from models.model import Model
class IdleController:
    def __init__(self, m):
        self.m = m # control dims

    def reset(self):
        pass

    def __call__(self, s, t):
        return np.zeros(self.m)

    def isFinished(self):
        return False
# /IdleController

class ReferenceTrackerController:
    '''
        Simply looks up the reference trajectory with which it was initialized
        and interpolates the state that should be next.
    '''
    def __init__(self, t_ref, s_ref, u_ref) -> None:
        self.t_ref = t_ref
        self.s_ref = s_ref
        self.u_ref = u_ref
        self.reset()

    def reset(self):
        self.t = 0
        self.is_finished = False
    #/

    # Note: this returns state, not control!
    def __call__(self, s, t):
        if self.isFinished():
            return np.zeros_like(self.u_ref[0])
        #/
        i = bisect.bisect_left(self.t_ref, t)
        if i >= len(self.t_ref):
            self.is_finished = True
            i = -1
        #/
        return self.s_ref[i]
    #/

    def isFinished(self):
        return self.is_finished
    #/
# /ReferenceTrackerController

class ILQRController:
    def __init__(self, mat_Ls, vec_ls, t_ref, s_ref, u_ref):
        self.mat_Ls = mat_Ls
        self.vec_ls = vec_ls
        self.t_ref = t_ref
        self.s_ref = s_ref
        self.u_ref = u_ref
        self.reset()
    #/

    def reset(self):
        self.t = 0
        self.is_finished = False
    #/

    def __call__(self, s, t):
        if self.isFinished():
            return np.zeros_like(self.u_ref[0])
        #/
        i = bisect.bisect_left(self.t_ref, t)
        if i >= len(self.s_ref) - 1:
            self.is_finished = True
            return np.zeros_like(self.u_ref[0])
        #/
        t_frac = (t - self.t_ref[i]) / (self.t_ref[i+1] - self.t_ref[i])
        s_ref_interp = self.s_ref[i] + t_frac * (self.s_ref[i+1] - self.s_ref[i])
        ds = s - s_ref_interp
        du = self.mat_Ls[i] @ ds + self.vec_ls[i]
        u = self.u_ref[i+1] + du
        return u
    #/

    def isFinished(self):
        return self.is_finished
    #/
# /ILQRController


class Robot:
    def __init__(self, model) -> None:
        self.model = model
        self.fsm_state = FsmState.IDLE

        # Trajectory
        self.s = None
        self.u = None
        self.t = None
        self.t0 = 0
        self.tf = 1
        self.idle_controller = IdleController(model.controlDim())
        self.controller = self.idle_controller
    #/__init__()
    
    def fsmTransition(self, fsm_state):
        if fsm_state != self.fsm_state:
            print('State transition: ', self.fsm_state, '->', fsm_state)
            self.fsm_state = fsm_state
        #/
    #/fsmTransition()

    def reset(self, s0):
        self.s = s0[np.newaxis,:]
        self.u = np.zeros((1,self.model.controlDim()))
        self.t = np.array([0])
        self.fsmTransition(FsmState.IDLE)
        self.controller = self.idle_controller
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
        s = self.model.applyControl(delta_t, s, u)
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

    def setController(self, controller):
        self.controller = controller
    #/

    def drive(self):
        self.s = self.s[0][np.newaxis,:]
        self.u = np.zeros((1,self.model.controlDim()))
        self.t = np.array([0])
        self.t0 = time.time()
        self.controller.reset()
        self.fsmTransition(FsmState.DRIVING)
    # /drive()

    def update(self):
        if self.fsm_state == FsmState.DRIVING:
            t = time.time() - self.t0
            if self.controller.__class__ == ReferenceTrackerController:
                s = self.controller(self.s[-1], self.t[-1])
                u = np.zeros(self.model.controlDim())
            else:
                dt = t - self.t[-1]
                s = self.applyControl(dt, self.s[-1], self.u[-1])
                u = self.controller(s, t)
            # /if-else
            self.t = np.append(self.t, t)
            self.s = np.append(self.s, s[np.newaxis,:], axis=0)
            self.u = np.append(self.u, u[np.newaxis,:], axis=0)
            if self.controller.isFinished():
                self.fsmTransition(FsmState.IDLE)
            #/
        # /if
    # /update()


    def currentPose(self):
        if self.s is None:
            return None
        else:
            return self.s[-1, 0:3]
        #/
    # /

    def renderCanonical(self, qpainter): # override
        raise NotImplementedError
    #/

    def render(self, qpainter):

        # to vehicle pose
        s = self.currentPose()
        if s is None:
            return
        #/

        original_transform = qpainter.worldTransform()
        x, y, θ = s
        qpainter.translate(x,y)
        qpainter.rotate(np.rad2deg(θ)) # qpainter rotates in degrees
        self.renderCanonical(qpainter)

        qpainter.setWorldTransform(original_transform)
        
        # Overlay elapsed time on top right
        if self.fsm_state == FsmState.DRIVING:
            time_str = '{:.2f} s'.format(self.t[-1])
            original_transform = qpainter.worldTransform()
            qpainter.translate(qpainter.device().width()-50, 20)
            qpainter.scale(1, -1)
            brush = QtGui.QBrush()
            brush.setColor(QtCore.Qt.red)
            qpainter.setBrush(brush)
            qpainter.drawText(0, 0, time_str)
            qpainter.setWorldTransform(original_transform)
        # /if

    # /render()

    def plotTrajectory(self, state_plot, control_plot): # override
        raise NotImplementedError
    #/
#/class Robot

