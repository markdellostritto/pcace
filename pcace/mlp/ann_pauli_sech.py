#****************************************************
# Import Statements
#****************************************************

import torch
import numpy as np
from typing import Dict, Union, Sequence, Callable, Optional
from ..ml import MLP
from ..tools import scatter_sum
from .force import get_outputs

#****************************************************
# Atomic Neural Network
#****************************************************

"""
    Atomic Neural Network - Pauli Repulsion
    Computes properties of a single atom using a neural network
    The properties are reduced over the total structure
    Finally, forces are computed as a gradient of the output
"""
class ANN_Pauli_Sech(torch.nn.Module):
    """
        n_in: input dimension of representation
        n_out: output dimension of target property (default: 1)
        n_hidden: size of hidden layers.
        activation: the activation function for each layer
        key_input: the key storing the NN input
        key_output: the key storing the NN output
        skip: whether to include a skip connection from input to output
        linout: whether the output layer has a linear activation function
    """
    # ==== initialization ====
    def __init__(
        self,
        # neural network
        n_in: int = None,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        activation: Callable = torch.nn.SiLU(),
        skip: bool = False,
        linout: bool = True,
        # constants
        ke: float = 14.3996454784562, # Coulomb's constant
        # radii
        radii: Dict[str,torch.tensor] = None,
        # keys - input/output
        key_input: Union[str, Sequence[int]] = 'node_feats',
        key_output_reduce: str = "z_tot",
        key_output_node: str = "z",
        # keys - energy/force
        key_energy  = "energy_pauli",
        key_forces  = "forces_pauli",
        key_virials = "virials_pauli",
        key_stress  = "stress_pauli",
        # calc flags
        calc_forces  = True,
        calc_virials = True,
        calc_stress  = True,
    ):
        # == init ==
        super().__init__()

        # == set nn data ==
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.activation = activation

        # == set radii ==
        self.ke = ke

        # == set radii ==
        self.radii = radii

        # == set keys - input/output ==
        self.key_input = key_input
        self.key_output_reduce = key_output_reduce
        self.key_output_node = key_output_node

        # == set keys - input/output ==
        self.key_energy  = key_energy
        self.key_forces  = key_forces
        self.key_virials = key_virials
        self.key_stress  = key_stress

        # == set calc flags ==
        self.calc_forces  = calc_forces
        self.calc_virials = calc_virials
        self.calc_stress  = calc_stress

        # == make the nn ==
        self.linout = linout
        self.outnet = MLP(
            n_in=self.n_in,
            n_out=self.n_out,
            n_hidden=self.n_hidden,
            activation=self.activation,
            linout=self.linout,
        )

        # == make the skip connection ==
        self.skip = skip
        if self.skip:
            self.linear_nn = MLP(
                self.n_in, 
                self.n_out,
                activation=None, 
            ) 
        else: self.linear_nn = None
        
    # ==== calculation ====
    def forward(self, 
        data: Dict[str, torch.Tensor],
        training: bool = None,
    ) -> Dict[str, torch.Tensor]:
        # == check features ==
        if not hasattr(self, "key_input") or self.key_input is None: self.key_input = "node_feats"
        if self.key_input not in data: raise ValueError(f"Input key {self.key_input} not found in data dictionary.")

        # == get features ==
        features = data[self.key_input]
        # reshape such that each node has its own entry
        features = features.reshape(features.shape[0], -1)

        # == predict atomic properties ==
        out_node = self.outnet(features)
        if self.skip: out_node += self.linear_nn(features)
        out_node=torch.squeeze(out_node)
        # == reduce the atomic properties ==
        out_reduce=scatter_sum(
            src=out_node,
            index=data["batch"],
            dim=0
        )
        
        # == reduce atomic data ==
        if self.key_output_node is not None: data[self.key_output_node] = out_node
        data[self.key_output_reduce] = out_reduce

        # == compute the energy ==
        #print("computing the energy")
        # compute edge lengths and vectors (normalized)
        vectors = data["positions"][data["edge_index"][1]] - data["positions"][data["edge_index"][0]] + data["shifts"]  # [n_edges, 3]
        edge_lengths = torch.linalg.norm(vectors, dim=-1, keepdim=False)  # [n_edges]
        # compute the pauli radius
        data["radius_pauli"]=torch.tensor(
            [self.radii[a.item()] for a in data["atomic_numbers"]],
            device=data["atomic_numbers"].device
        )
        # compute the interaction coefficients
        alpha=1.0/(data["radius_pauli"]*data["radius_pauli"])
        gamma=2.0*alpha[data["edge_index"][0]]*alpha[data["edge_index"][1]]\
            /(alpha[data["edge_index"][0]]+alpha[data["edge_index"][1]])
        amp=torch.pow(
            2.0*torch.sqrt(alpha[data["edge_index"][0]]*alpha[data["edge_index"][1]])\
            /(alpha[data["edge_index"][0]]+alpha[data["edge_index"][1]])
        ,3)
        # compute the edge energy
        energy_edge = self.ke\
            *data[self.key_output_node][data["edge_index"][0]]\
            *data[self.key_output_node][data["edge_index"][1]]\
            *amp/edge_lengths\
            *torch.cosh(torch.sqrt(gamma)*edge_lengths).pow(-2.0)
        # compute the node energy
        n_nodes = data["positions"].shape[0]
        energy_node = 0.5*scatter_sum(
            src=energy_edge, 
            index=data["edge_index"][1],
            dim=0, 
            dim_size=n_nodes
        )
        # compute the structure energy
        data[self.key_energy]=scatter_sum(
            src=energy_node,
            index=data["batch"],
            dim=0
        )
        
        # == compute the forces ==
        forces, virials, stress = get_outputs(
            energy = data[self.key_energy],
            positions = data['positions'],
            displacement = data.get('displacement', None),
            cell = data.get('cell', None),
            training=training,
            compute_force = self.calc_forces,
            compute_virials = self.calc_virials,
            compute_stress = self.calc_stress
        )
        data[self.key_forces] = forces
        if self.key_virials is not None:
            data[self.key_virials] = virials
        if self.key_stress is not None:
            data[self.key_stress] = stress
        
        # == return ==
        return data

    # ==== output ====
    def __repr__(self):
        return (
            f"\n==============================================\n"
            f"{self.__class__.__name__}\n"
            # constants
            f"ke = {self.ke}\n"
            # calc flags
            f"calc_forces = {self.calc_forces}\n"
            f"calc_virials = {self.calc_virials}\n"
            f"calc_stress = {self.calc_stress}\n"
            # keys - input/output
            f"key_input = {self.key_input}\n"
            f"key_output_reduce = {self.key_output_reduce}\n"
            f"key_output_node = {self.key_output_node}\n"
            # keys - energy/force
            f"key_energy = {self.key_energy}\n"
            f"key_forces = {self.key_forces}\n"
            f"key_virials = {self.key_virials}\n"
            f"key_stress = {self.key_stress}\n"
            # radii
            f"radii = {self.radii}\n"
            # neural network
            f"n_in = {self.n_in}\n"
            f"n_out = {self.n_out}\n"
            f"n_hidden = {self.n_hidden}\n"
            f"activation = {self.activation}\n"
            f"skip = {self.skip}\n"
            f"linout = {self.linout}\n"
            # neural nets
            f"{self.outnet}\n"
            f"{self.linear_nn}\n"
            f"**********************************************"
        )