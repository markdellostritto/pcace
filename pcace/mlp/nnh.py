#****************************************************
# Import Statements
#****************************************************

import torch
from typing import Dict

#****************************************************
# Neural Network Hamiltonian 
#****************************************************

class NNH(torch.nn.Module):
    # ==== initialization ====
    def __init__(
        self,
        # representation
        rep: torch.nn.ModuleList = None, 
        # atomic neural network
        ann: torch.nn.ModuleList = None, 
    ):
        super().__init__()
        # set representation
        self.rep = rep
        # atomic neural network
        self.ann = ann
        
    # ==== calculation ====
    def forward(
        self, 
        data: Dict[str, torch.Tensor],
        training = True,
        compute_forces = True,
        compute_stress = True,
        compute_virials = True,
        compute_forces_edge = False,
    ):
        # == check compute flags ==
        if not (compute_forces ^ compute_forces_edge):
            raise ValueError("One must either compute forces based on positions OR vectors, not both.")
        if compute_forces and "positions" not in data:
            raise ValueError("Need positions to compute total atomic forces.")
        if compute_forces_edge and "vectors" not in data:
            raise ValueError("Need vectors to compute edge forces.")
        
        # == compute the energy ==
        #print("computing energy")
        # compute representation
        data = self.rep(data)
        # compute ann
        data = self.ann(data,
            training,
            compute_forces,
            compute_stress,
            compute_virials,
            compute_forces_edge,
        )
                
        # == return the data ==
        return data
        
    # ==== output ====
    def __repr__(self):
        return (
            f"\n=========================================================\n"
            f"{self.__class__.__name__}\n"
            # representation
            f"rep = {self.rep}\n"
            # atomic neural network
            f"ann = {self.ann}\n"
            f"---------------------------------------------------------\n"
            f"========================================================="
        )