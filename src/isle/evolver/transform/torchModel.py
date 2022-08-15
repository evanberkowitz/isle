import torch
import numpy as np
from .transform import Transform
from ... import CDVector

class TorchTransform(Transform):
    def __init__(self, torchModel, action, desired_shape):
        super(TorchTransform, self).__init__()
        self.torchModel = torchModel
        self.action = action
        self.shape = desired_shape
        self.volume = 1
        for N in desired_shape:
            self.volume*=N

    def forward(self, phi, actVal = None):
        if not isinstance(phi,torch.Tensor):
            phi = torch.from_numpy(np.array(phi)).reshape(self.shape)

        with torch.no_grad():
            out,logDetJ = self.torchModel(phi)
            out =  CDVector(out.reshape(self.volume).numpy())
            logDetJ = logDetJ.item()

        actVal = self.action.eval(out)

        return out, actVal, logDetJ

    def backward(self, phi, jacobian=False):
        raise NotImplementedError(
            "To not explicitly invert a torch model, this is not implemented"
        )

    def save(self,h5group,manager):
        pass
        #h5group['model dict'] = self.torchModel.state_dict()

    @classmethod
    def fromH5(cls, h5group, manager, action, lattice, rng, TorchModelConstructor, *args, **kwargs):
        raise NotImplementedError("torch does not provide a h5 interface yet")
        model = TorchModelConstructor(*args,**kwargs)
        model.load_state_dict(h5group['model dict'][()])

        return cls(model,action)

class TorchTransformSlowLogDet(Transform):
    def __init__(self, torchModel, action):
        super(TorchTransformSlowLogDet, self).__init__()
        self.torchModel = torchModel
        self.action = action

    def forward(self, phi, actVal = None):
        Nt = 16
        Nx = 2
        if not isinstance(phi,torch.Tensor):
            phi = torch.from_numpy(np.array(phi)).reshape((Nt,Nx))

        with torch.no_grad():
            out,logDetJ = self.torchModel(phi)
            out =  CDVector(out.reshape(Nt*Nx).numpy())
            logDetJ = logDetJ.item()

        actVal = self.action.eval(out)

        return out, actVal, logDetJ

    def calcLogDetJ(self, phi):
        if not isinstance(phi,torch.Tensor):
            phi = torch.from_numpy(np.array(phi))

        return -self.torchModel.calcLogDetJ(phi).item()

    def backward(self, phi, jacobian=False):
        raise NotImplementedError(
            "To not explicitly invert a torch model, this is not implemented"
        )

    def save(self,h5group,manager):
        pass
        #h5group['model dict'] = self.torchModel.state_dict()

    @classmethod
    def fromH5(cls, h5group, manager, action, lattice, rng, TorchModelConstructor, *args, **kwargs):
        raise NotImplementedError("torch does not provide a h5 interface yet")
        model = TorchModelConstructor(*args,**kwargs)
        model.load_state_dict(h5group['model dict'][()])

        return cls(model,action)

