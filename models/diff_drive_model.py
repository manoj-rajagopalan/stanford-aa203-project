import numpy as np
from models.model import Model

class DiffDriveModel(Model):
    def __init__(self, r, L):
        self.r = r # wheel radius
        self.L = L  # baseline
    # /__init__()

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
        return 'ω_l', 'ω_r'
    #/

    def parameters(self):
        return (self.r, self.L)
    #/

    def dynamics(self, t, s, u):
        _, _, θ = s
        ω_l, ω_r = u
        v = (self.r/2) * (ω_r + ω_l)
        x_dot = v * np.cos(θ)
        y_dot = v * np.sin(θ)
        θ_dot = (self.r/self.L) * (ω_r - ω_l)
        return np.array([x_dot, y_dot, θ_dot])
    # /dynamics()

    def dynamicsJacobianWrtState(self, s, u):
        J_s = np.zeros((self.stateDim(), self.stateDim()))
        θ = s[2]
        v = self.r/2 * np.sum(u)
        J_s[0,2] = -v * np.sin(θ)
        J_s[1,2] =  v * np.cos(θ)
        return J_s
    # /dynamicsJacobianWrtState()

    def dynamicsJacobianWrtControl(self, s, u):
        J_u = np.zeros((self.stateDim(), self.controlDim()))
        θ = s[2]
        J_u[0,:].fill(self.r/2 * np.cos(θ))
        J_u[1,:].fill(self.r/2 * np.sin(θ))
        J_u[2,0] = -self.r / self.L
        J_u[2,1] =  self.r / self.L
        return J_u
    # /dynamicsJacobianWrtControl()

# /class DiffDriveModel
