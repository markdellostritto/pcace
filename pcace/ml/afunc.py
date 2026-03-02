import numpy as np
import torch

class SoftPlusZero(torch.nn.Module):
    # ==== initialization ====
    def __init__(self):
        super(SoftPlusZero, self).__init__()

    # ==== calculation ====
    def forward(self, x):
        return torch.log1p(torch.exp(x))-np.log(2.0)
        
class SquarePlusZero(torch.nn.Module):
    # ==== initialization ====
    def __init__(self):
        super(SquarePlusZero, self).__init__()

    # ==== calculation ====
    def forward(self, x):
        return 0.5*(x-1.0+torch.sqrt(1.0+x*x))
    
class IERFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, c=1.0):
        # radial pi
        rad_pi = np.sqrt(np.pi)
        # input
        z = input.detach().cpu().numpy()
        # compute exponential
        fexp = np.exp(-z * z)
        zs = np.sign(z)
        # compute error function
        t = 1.0 / (1.0 + 0.3275911 * np.abs(z))
        poly = (
            0.254829592 + 
            t * (-0.284496736 + 
            t * (1.421413741 + 
            t * (-1.453152027 + 
            t * 1.061405429)))
        )
        ferf = zs * (1.0 - t * poly * fexp)
        # activation
        a = 0.5 * (z * ferf + (fexp + z * rad_pi - 1.0) / rad_pi)
        output = torch.from_numpy(a).to(input.device).type(input.dtype)
        # derivative
        d = 0.5 * (ferf + 1.0)
        derivative = torch.from_numpy(d).to(input.device).type(input.dtype)
        # context
        ctx.save_for_backward(derivative)
        # return ouput
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        derivative, = ctx.saved_tensors
        return grad_output * derivative, None

class IERF(torch.nn.Module):
    def __init__(self, c=1.0):
        super(IERF, self).__init__()
        self.c = c
    
    def forward(self, x):
        return IERFFunction.apply(x, self.c)

