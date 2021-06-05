import numpy as np
import scipy.integrate

class Model:
    def stateDim(self): # override
        raise NotImplementedError
    #/

    def stateNames(self): # override
        raise NotImplementedError
    #/

    def controlDim(self): # override
        raise NotImplementedError
    #/

    def controlNames(self): # override
        raise NotImplementedError
    #/

    def parameters(self): # override
        raise NotImplementedError
    #/

    def dynamics(t, s, u): # override
        raise NotImplementedError
    #/

    def dynamicsJacobianWrtState(self, s, u): # override
        raise NotImplementedError
    #/

    def dynamicsJacobianWrtControl(self, s, u): # override
        raise NotImplementedError
    #/

    def applyControl(self, delta_t, s, u):
        s = scipy.integrate.odeint(self.dynamics,
                                   s,
                                   np.array([0, delta_t]),
                                   args=(u,),
                                   tfirst=True)[1]
        return s
    # /applyControl()

    def generateTrajectory(self, t, s0, u):
        N = len(t) - 1
        assert N == len(u)
        s = np.zeros((N+1, self.stateDim()))
        s[0] = s0
        for n in range(N):
            dt = t[n+1] - t[n]
            s[n+1] = self.applyControl(dt, s[n], u[n])
        # /for n
        return s
    # /generateTrajectory()

# /class Model
