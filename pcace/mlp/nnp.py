#****************************************************
# Import Statements
#****************************************************

import torch
from typing import Dict
from .force import get_symmetric_displacement

#****************************************************
# Neural Network Potential 
#****************************************************

class NNP(torch.nn.Module):
    # ==== initialization ====
    def __init__(
        self,
        # representation
        rep: torch.nn.Module = None, 
        # atomic neural network
        annl: torch.nn.ModuleList = None, 
        # calc flags
        calc_forces = True,
        calc_virials = True,
        calc_stress = True,
        # keys - energy/force
        key_energy = "energy",
        key_forces = "forces",
        key_stress = "stress",
    ):
        super().__init__()
        # set representation
        self.rep = rep
        # atomic neural network
        self.annl = annl
        # set calc flags
        self.calc_forces  = calc_forces
        self.calc_virials = calc_virials
        self.calc_stress  = calc_stress
        for ann in self.annl:
            ann.calc_force   = self.calc_forces
            ann.calc_virials = self.calc_virials
            ann.calc_stress  = self.calc_stress
        # set keys
        self.key_energy=key_energy
        self.key_forces=key_forces
        self.key_stress=key_stress

    # ==== calculation ====
    def forward(
        self, 
        data: Dict[str, torch.Tensor],
        training = True,
    ):
        # == set gradients required ==
        data["positions"].requires_grad_(True)

        # == get the number of graphs ==
        try:
            data["num_graphs"]=data["ptr"].numel()-1
        except:
            data["num_graphs"]=1
        
        # == get symmetric displacement ==
        #print("get_symmetric_displacement")
        if(self.calc_stress or self.calc_virials):
            (
                data["positions"],
                data["shifts"],
                data["displacement"],
                data["cell"]
            )=get_symmetric_displacement(
                data["positions"],
                data["unit_shifts"],
                data["cell"],
                data["edge_index"],
                data["num_graphs"],
                data["batch"]
            )
        else:
            data["displacement"] = None
        
        # == compute the representation ==
        #print("computing representation")
        data = self.rep(data)

        # == compute the energy ==
        #print("computing energy")
        energies = []
        forces = []
        stress = []
        for ann in self.annl:
            data = ann(data,training)
            energies.append(data[ann.key_energy])
            forces.append(data[ann.key_forces])
            stress.append(data[ann.key_stress])
        data[self.key_energy]=torch.stack(energies).sum(0)
        data[self.key_forces]=torch.stack(forces).sum(0)
        data[self.key_stress]=torch.stack(stress).sum(0)

        # == return the data ==
        return data
        
    # ==== output ====
    def __repr__(self):
        return (
            f"\n==============================================\n"
            f"{self.__class__.__name__}\n"
            # calc flags
            f"calc_forces = {self.calc_forces}\n"
            f"calc_virials = {self.calc_virials}\n"
            f"calc_stress = {self.calc_stress}\n"
            # keys - energy/force
            f"key_energy = {self.key_energy}\n"
            f"key_forces = {self.key_forces}\n"
            f"key_stress = {self.key_stress}\n"
            # representation
            f"rep = {self.rep}\n"
            # atomic neural network
            f"ann_len = {len(self.annl)}\n"
            f"annl = {self.annl}\n"
            f"**********************************************"
        )