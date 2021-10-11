import torch
import numpy as np

from .transform import Transform
from ... import CDVector

class TorchTransform(Transform):
    def __init__(self, torchModel, action):
        super(TorchTransform, self).__init__()
        self.torchModel = torchModel
        self.action = action

    def forward(self, phi, actVal):
        out = self.torchModel(
            torch.from_numpy(np.array(phi))
        )
        logDetJ = self.torchModel.calcLogDetJ(out)


        out = CDVector(out.detach().numpy())
        actVal = self.action.eval(out)

        return out, actVal, logDetJ

    def calcLogDetJ(self, phi):
        if not isinstance(phi,torch.Tensor):
            phi = torch.from_numpy(np.array(phi))
        return self.torchModel.calcLogDetJ(phi)

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
