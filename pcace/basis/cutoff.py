#****************************************************
# Import Statements
#****************************************************

import torch

__all__ = ["CutoffStep", "CutoffCos", "CutoffPoly3"]

#****************************************************
# Cutoff Functions
#****************************************************

# ==== Step ====

"""
    Step function cutoff
    @member rc - cutoff distance
    Cutoff function - step function - zero for r>rc
    A true Heaviside step function would preclude the calculation of
        gradients, and so here we use a sigmoid function with a large
        constant to yield a sharp transition to zero at rc.
"""
class CutoffStep(torch.nn.Module):
    # initialization
    def __init__(self, rc: float):
        super().__init__()
        self.register_buffer("rc", torch.tensor(rc))
    # calculation
    def forward(self, dr: torch.Tensor) -> torch.Tensor:
        return 1.0/(1.0+torch.exp(1000.0*(dr-self.rc)))*(dr<self.rc).float()
    # output
    def __repr__(self):
        return f"{self.__class__.__name__}(rc={self.rc})"
    
# ==== Cos ====

"""
    Cosine function cutoff
    @member rc - cutoff distance
    Cutoff function - cosine
    Cosine cutoff function that smoothly transitions from 1 at 0 to 0 at rc.
"""
class CutoffCos(torch.nn.Module):
    # initialization
    def __init__(self, rc: float):
        super().__init__()
        self.register_buffer("rc", torch.tensor(rc))
    # calculation
    def forward(self, dr: torch.Tensor) -> torch.Tensor: # [...,1]
        return 0.5*(torch.cos(dr*torch.pi/self.rc)+1.0)*(dr<self.rc).float() # [...,1]
    # output
    def __repr__(self):
        return f"{self.__class__.__name__}(rc={self.rc})"
    
# ==== Poly3 ====

"""
    Polynomial function cutoff
    @member rc - cutoff distance
    Cutoff function - polynomial
    Polynomial cutoff function that smoothly transitions from 1 at 0 to 0 at rc.
    The polynomial is of order three, allowing for gradients necessary for 
        training and evaluation.  Note that the shape of this polynomial very closely
        follows the shape of the cosine cutoff function.
"""
class CutoffPoly3(torch.nn.Module):
    # initialization
    def __init__(self, rc: float):
        super().__init__()
        self.register_buffer("rc", torch.tensor(rc))
    # calculation
    def forward(self, dr: torch.Tensor) -> torch.Tensor: # [...,1]
        return (1.0+(2.0*(dr/self.rc)-3.0)*(dr/self.rc)**2)*(dr<self.rc).float() # [...,1]
    # output
    def __repr__(self):
        return f"{self.__class__.__name__}(rc={self.rc})"
