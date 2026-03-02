#****************************************************
# Import Statements
#****************************************************

import torch

#****************************************************
# Loss Functions
#****************************************************

class LossMSE(torch.nn.Module):
    #==== initialization ====
    def __init__(self, a: float=1.0):
        super().__init__()
        self.register_buffer("a", torch.tensor(a, dtype=torch.get_default_dtype()))
    #==== calculation ====
    def forward(self, output, target):
        loss = 0.5*torch.mean((output - target)**2)
        return loss
    
class LossHuber(torch.nn.Module):
    #==== initialization ====
    def __init__(self, a: float=1.0e-3):
        super().__init__()
        self.register_buffer("a", torch.tensor(a, dtype=torch.get_default_dtype()))
    #==== calculation ====
    def forward(self, output, target):
        d=(output-target)*1.0/self.a
        loss = torch.mean(self.a*self.a*(torch.sqrt(1.0+d**2)-1.0))
        return loss
    
class LossAsinh(torch.nn.Module):
    #==== initialization ====
    def __init__(self, a: float=1.0e-3):
        super().__init__()
        self.register_buffer("a", torch.tensor(a, dtype=torch.get_default_dtype()))
    #==== calculation ====
    def forward(self, output, target):
        d=(output-target)*1.0/self.a
        sqrtf = torch.sqrt(1.0+d**2)
        logf = torch.log(d+sqrtf)
        loss = torch.mean(self.a*self.a*(1.0-sqrtf+d*logf))
        return loss