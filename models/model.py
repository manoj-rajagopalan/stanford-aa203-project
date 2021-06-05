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

# /class Model
