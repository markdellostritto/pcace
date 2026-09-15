#****************************************************
# Import Statements
#****************************************************

import torch
from typing import Dict
from .force import get_symmetric_displacement

#****************************************************
# Neural Network Surface 
#****************************************************

class NNS(torch.nn.Module):
    # ==== initialization ====
    def __init__(
        self,
        # hamiltonian list
        nnhl: torch.nn.ModuleList = None, 
        # keys - energy/force
        key_energy = "energy_nns",
        key_forces = "forces_nns",
        key_virials = "virials_nns",
        key_stress = "stress_nns",
        key_forces_edge = "forces_edge_nns",
    ):
        super().__init__()
        # hamiltonian list
        self.nnhl = nnhl
        self.rc = 0.0
        for nnh in nnhl:
            if(nnh.rep.cutoff.rc>self.rc): self.rc=nnh.rep.cutoff.rc
        # set keys
        self.key_energy = key_energy
        self.key_forces = key_forces
        self.key_virials = key_virials
        self.key_stress = key_stress
        self.key_forces_edge = key_forces_edge

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
        
        # == set gradients required ==
        if "positions" in data: data["positions"].requires_grad_(True)
        if "vectors" in data: data["vectors"].requires_grad_(True)

        # == get the number of graphs ==
        try:
            data["num_graphs"] = data["ptr"].numel()-1
        except:
            data["num_graphs"] = 1
        #print("num_graphs = ",data["num_graphs"])
        
        # == set the batch ==
        if data["batch"] == None: 
            n_nodes = data["atomic_numbers"].shape[0]
            data["batch"] = torch.zeros(n_nodes,dtype=torch.int64,device=data["atomic_numbers"].device)
        
        # == get symmetric displacement ==
        #print("get_symmetric_displacement")
        if(compute_stress or compute_virials):
            (
                data["positions"],
                data["shifts"],
                data["displacement"],
                data["cell"]
            ) = get_symmetric_displacement(
                data["positions"],
                data["unit_shifts"],
                data["cell"],
                data["edge_index"],
                data["num_graphs"],
                data["batch"]
            )
        else:
            data["displacement"] = None
        
        # == compute the energy ==
        #print("computing energy")
        energies = []
        forces = []
        stress = []
        virials = []
        forces_edge = []
        for nnh in self.nnhl:
            if(nnh.ann.weight>0.0):
                data = nnh(data,
                    training,
                    compute_forces,
                    compute_stress,
                    compute_virials,
                    compute_forces_edge,
                )
                energies.append(data[nnh.ann.key_energy])
                if compute_forces: forces.append(data[nnh.ann.key_forces])
                if compute_stress: stress.append(data[nnh.ann.key_stress])
                if compute_virials: virials.append(data[nnh.ann.key_virials])
                if compute_forces_edge: forces_edge.append(data[nnh.ann.key_forces_edge])
        data[self.key_energy]=torch.stack(energies).sum(0)
        if compute_forces: data[self.key_forces]=torch.stack(forces).sum(0)
        if compute_stress: data[self.key_stress]=torch.stack(stress).sum(0)
        if compute_virials: data[self.key_virials]=torch.stack(virials).sum(0)
        if compute_forces_edge: data[self.key_forces_edge]=torch.stack(forces_edge).sum(0)

        # == return the data ==
        return data
        
    # ==== output ====
    def __repr__(self):
        return (
            f"\n=========================================================\n"
            f"{self.__class__.__name__}\n"
            # keys - energy/force
            f"key_energy = {self.key_energy}\n"
            f"key_forces = {self.key_forces}\n"
            f"key_stress = {self.key_stress}\n"
            f"key_virials = {self.key_virials}\n"
            f"key_forces_edge = {self.key_forces_edge}\n"
            # neural network hamiltonians
            f"rc   = {self.rc}\n"
            f"nnhl = {self.nnhl}\n"
            f"---------------------------------------------------------\n"
            f"========================================================="
        )