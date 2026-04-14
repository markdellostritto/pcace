#****************************************************
# Import Statements
#****************************************************

import torch

__all__ = [
    "RadialBesselJ",
    "RadialBesselY",
    "RadialGaussian",
    "RadialLogistic",
    "RadialLogCosh",
    "RadialExp",
    "RadialSoftPlus",
    "RadialChebyshev"
]

#****************************************************
# Radial Functions
#****************************************************

"""
    BesselJ radial functions
    @member rc - cutoff distance
    @member nr - number of radial functions
    @member pf - BesselJ prefactor 
    @member weights - weights associated with each BesselJ function (zero crossings)
    @member train - whether to train the parameters
    A series of BesselJ functions with their origin at 0 (sinc functions)
    The weights determine the number of zero-crossings between 0 and rc.
    Note that rc is required in the definition of the function in order
        to scale the weights so that integer weights yield integer crossings
        before rc, the function itself does not go to zero at rc.
    The fluctuation of the sinc functino between 0 and rc gives multiple maxima
        and minima.  Several of these functions together thus yield an effective
        radial basis, allowing for discrimination of important signals at all
        distances between 0 and rc.
"""
class RadialBesselJ(torch.nn.Module):
    # ==== initialization ====
    def __init__(self, rc: float, nr: int, train=False):
        super().__init__()
        # set the number of functions
        self.nr=nr
        # set the weights
        weights = torch.pi/rc * torch.linspace(
            start=1.0,end=nr,steps=nr
        )
        if train: 
            self.weights=torch.nn.Parameter(weights,requires_grad=True)
        else: 
            self.register_buffer("weights", weights)
        # set constants
        self.register_buffer(
            "rc", torch.tensor(rc, dtype=torch.get_default_dtype())
        )
        
    # ==== calculation ====
    def forward(self, dr: torch.Tensor) -> torch.Tensor:  # [..., 1]
        return torch.sin(self.weights*dr)/dr
        
    # ==== output ====
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(rc={self.rc}, nr={len(self.weights)}, "
            f"train={self.weights.requires_grad})"
        )

"""
    BesselY radial functions
    @member rc - cutoff distance
    @member nr - number of radial functions
    @member pf - BesselY prefactor 
    @member weights - weights associated with each BesselY function (zero crossings)
    @member train - whether to train the parameters
    A series of BesselY functions with their origin at 0 (cosc functions)
    The weights determine the number of zero-crossings between 0 and rc.
    Note that rc is required in the definition of the function in order
        to scale the weights so that integer weights yield integer crossings
        before rc, the function itself does not go to zero at rc.
    The fluctuation of the cosc functino between 0 and rc gives multiple maxima
        and minima.  Several of these functions together thus yield an effective
        radial basis, allowing for discrimination of important signals at all
        distances between 0 and rc.
"""
class RadialBesselY(torch.nn.Module):
    # ==== initialization ====
    def __init__(self, rc: float, nr: int, train=False):
        super().__init__()
        # set the number of functions
        self.nr=nr
        # set the weights
        weights = torch.pi/rc * torch.linspace(
            start=1.0,end=nr,steps=nr
        )
        if train: 
            self.weights=torch.nn.Parameter(weights,requires_grad=True)
        else: 
            self.register_buffer("weights", weights)
        # set constants
        self.register_buffer(
            "rc", torch.tensor(rc, dtype=torch.get_default_dtype())
        )
        
    # ==== calculation ====
    def forward(self, dr: torch.Tensor) -> torch.Tensor:  # [..., 1]
        return torch.cos(self.weights*dr)/dr
        
    # ==== output ====
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(rc={self.rc}, nr={len(self.weights)}, "
            f"train={self.weights.requires_grad})"
        )

"""
    Gaussian radial functions
    @member rc - cutoff distance
    @member nr - number of radial functions
    @member weights - weights associated with each Gaussian function (positions)
    @member train - whether to train the parameters
"""
class RadialGaussian(torch.nn.Module):
    # ==== initialization ====
    def __init__(self, rc: float, nr=8, train=False):
        super().__init__()
        # set the number of functions
        self.nr=nr
        # set the weights
        offsets = torch.linspace(0.1, rc, nr)
        widths = torch.linspace(1.0,nr+1.0,nr)*0.1
        # set train
        if train: 
            self.offsets=torch.nn.Parameter(offsets,requires_grad=True)
            self.widths=torch.nn.Parameter(widths,requires_grad=True)
        else: 
            self.register_buffer("offsets", widths)
            self.register_buffer("widths", widths)
        # set constants
        self.register_buffer(
            "rc", torch.tensor(rc, dtype=torch.get_default_dtype())
        )
        
    # ==== calculation ====
    def forward(self, dr: torch.Tensor) -> torch.Tensor:  # [..., 1]
        return torch.exp(-0.5*((dr-self.offsets)/self.widths)**2) # [...,nr]
    
    # ==== output ====
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(rc={self.rc}, nr={len(self.widths)}, "
            f"train={self.widths.requires_grad})"
        )
    
"""
    Logistic radial functions
    @member rc - cutoff distance
    @member nr - number of radial functions
    @member weights - weights associated with each Logistic function (positions)
    @member train - whether to train the parameters
"""
class RadialLogistic(torch.nn.Module):
    # ==== initialization ====
    def __init__(self, rc: float, nr=8, train=False):
        super().__init__()
        # set the number of functions
        self.nr=nr
        # set the weights
        offsets = torch.linspace(0.1, rc, nr)
        widths = torch.ones(nr)
        # set train
        if train: 
            self.offsets=torch.nn.Parameter(offsets,requires_grad=True)
            self.widths=torch.nn.Parameter(widths,requires_grad=True)
        else: 
            self.register_buffer("offsets", widths)
            self.register_buffer("widths", widths)
        # set constants
        self.register_buffer(
            "rc", torch.tensor(rc, dtype=torch.get_default_dtype())
        )
        
    # ==== calculation ====
    def forward(self, dr: torch.Tensor) -> torch.Tensor:  # [..., 1]
        return 1.0/torch.cosh(self.wdidths*(dr-self.offsets))**2 # [...,nr]
    
    # ==== output ====
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(rc={self.rc}, nr={len(self.widths)}, "
            f"train={self.widths.requires_grad})"
        )

"""
    LogCosh radial functions
    @member rc - cutoff distance
    @member nr - number of radial functions
    @member weights - weights associated with each function (positions)
    @member train - whether to train the parameters
"""
class RadialLogCosh(torch.nn.Module):
   # ==== initialization ====
    def __init__(self, rc: float, nr: int, train=False):
        super().__init__()
        # set the number of functions
        self.nr=nr
        # set the weights
        offsets = torch.ones(nr)*0.1
        widths = torch.linspace(0.1, rc, nr)
        # set train
        if train: 
            self.offsets=torch.nn.Parameter(offsets,requires_grad=True)
            self.widths=torch.nn.Parameter(widths,requires_grad=True)
        else: 
            self.register_buffer("offsets", widths)
            self.register_buffer("widths", widths)
        # set constants
        self.register_buffer(
            "rc", torch.tensor(rc, dtype=torch.get_default_dtype())
        )
        
    # ==== calculation ====
    def forward(self, dr: torch.Tensor) -> torch.Tensor:  # [..., 1]
        return torch.log1p(torch.exp(-self.widths*(dr-self.offsets)))
    
    # ==== output ====
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(rc={self.rc}, nr={len(self.widths)}, "
            f"train={self.widths.requires_grad})"
        )

"""
    Exponential radial functions
    @member rc - cutoff distance
    @member nr - number of radial functions
    @member weights - weights associated with each function (positions)
    @member train - whether to train the parameters
"""
class RadialExp(torch.nn.Module):
   # ==== initialization ====
    def __init__(self, rc: float, nr: int, train=False):
        super().__init__()
        # set the number of functions
        self.nr=nr
        # set the weights
        weights = torch.pi/rc * torch.linspace(
            start=1.0,end=nr,steps=nr
        )
        # set train
        if train: 
            self.weights=torch.nn.Parameter(weights,requires_grad=True)
        else: 
            self.register_buffer("weights", weights)
        # set constants
        self.register_buffer(
            "rc", torch.tensor(rc, dtype=torch.get_default_dtype())
        )
        
    # ==== calculation ====
    def forward(self, dr: torch.Tensor) -> torch.Tensor:  # [..., 1]
        return torch.exp(-self.weights*dr)
    
    # ==== output ====
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(rc={self.rc}, nr={len(self.weights)}, "
            f"train={self.weights.requires_grad})"
        )

class RadialSoftPlus(torch.nn.Module):
    # ==== initialization ====
    def __init__(self, rc: float, nr: int, train=False):
        super().__init__()
        # set the number of functions
        self.nr=nr
        # set the weights
        weights = torch.pi/rc * torch.linspace(
            start=1.0,end=nr,steps=nr
        )
        # set train
        if train: 
            self.weights=torch.nn.Parameter(weights,requires_grad=True)
        else: 
            self.register_buffer("weights", weights)
        # set constants
        self.register_buffer(
            "rc", torch.tensor(rc, dtype=torch.get_default_dtype())
        )
        
    # ==== calculation ====
    def forward(self, dr: torch.Tensor) -> torch.Tensor:  # [..., 1]
        return self.weights*dr/(1.0+torch.exp(self.weights*dr))
        
    # ==== output ====
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(rc={self.rc}, nr={len(self.weights)}, "
            f"train={self.weights.requires_grad})"
        )

class RadialChebyshev(torch.nn.Module):
    # ==== initialization ====
    def __init__(self, rc: float, nr: int, train=False):
        super().__init__()
        # set the number of functions
        self.nr=nr
        # set the weights
        weights = torch.linspace(
            start=1.0,end=nr,steps=nr
        )
        # set train
        if train: 
            self.weights=torch.nn.Parameter(weights,requires_grad=True)
        else: 
            self.register_buffer("weights", weights)
        # set constants
        self.register_buffer(
            "rc", torch.tensor(rc, dtype=torch.get_default_dtype())
        )
        
    # ==== calculation ====
    def forward(self, dr: torch.Tensor) -> torch.Tensor:  # [..., 1]
        return torch.cos(self.weights*torch.acos(2.0*(dr/self.rc)-1.0))
        
    # ==== output ====
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(rc={self.rc}, nr={len(self.weights)}, "
            f"train={self.weights.requires_grad})"
        )
