#****************************************************
# Import Statements
#****************************************************

import itertools
import numpy as np
import torch
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
class ANN_LDamp_Long(torch.nn.Module):
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
        weight: float = 1.0,
        # elements
        radii: Dict[str,torch.tensor] = None,
        # kspace
        rc: float = 0.0,
        prec: float = 1.0e-6,
        # keys - input/output
        key_input: Union[str, Sequence[int]] = 'node_feats',
        key_output_reduce: str = "c_tot",
        key_output_node: str = "c",
        # keys - energy/force
        key_energy  = "energy_ldamp_long",
        key_forces  = "forces_ldamp_long",
        key_virials = "virials_ldamp_long",
        key_stress  = "stress_ldamp_long",
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
        self.weight = weight

        # == set kspace ==
        self.rc = rc
        self.prec = prec
        
        # == set elements ==
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
        out_node = torch.squeeze(out_node)
        # == reduce the atomic properties ==
        out_reduce=scatter_sum(
            src=out_node,
            index=data["batch"],
            dim=0
        )
        
        # == reduce atomic data ==
        if self.key_output_node is not None: data[self.key_output_node] = out_node
        data[self.key_output_reduce] = out_reduce

        # == init ==
        nGraphs = torch.unique(data["batch"],return_counts=False).shape[0]
        # compute sum over charge, charge squared
        cs = scatter_sum(src=out_node,index=data["batch"],dim=0)
        c2s = scatter_sum(src=out_node*out_node,index=data["batch"],dim=0)
        # compute reciprocal lattice
        cellR = data['cell'].view(-1, 3, 3)
        cellK = torch.transpose(2.0*np.pi*torch.linalg.inv(cellR),1,2)
        vol = torch.linalg.det(cellR)
        # compute convergence constant
        kAlphaG = (1.35-0.15*np.log(self.prec))/self.rc
        kAlpha = torch.tensor([kAlphaG for _ in range(nGraphs)],device=data["batch"].device)
        #print("kAlpha = ",kAlpha)
        # compute reciprocal lattice points
        nk = [8,8,8] # approximation
        kpoints = []
        for ix,iy,iz in itertools.product(range(-nk[0],nk[0]+1), range(-nk[1],nk[1]+1), range(-nk[2],nk[2]+1)):
            if(np.max(np.abs(np.array([ix,iy,iz])))): kpoints.append([ix,iy,iz])
        kpoints = torch.tensor(kpoints,device=data["batch"].device)
        
        # == compute the energy - constant term ==
        #print("computing the energy - constant term")
        ec = (-1.0/6.0*np.pi**(3.0/2.0)/vol*kAlpha**3*cs*cs+1.0/12.0*kAlpha**6*c2s)*self.weight
        #print("ec = ",ec)
        
        # == compute the energy - rspace ==
        #print("computing the energy - rspace term")
        # compute the vdw radius
        data["radius_vdw"]=torch.tensor(
            [self.radii[a.item()] for a in data["atomic_numbers"]],
            device=data["atomic_numbers"].device
        )
        rvdw=0.5*(data["radius_vdw"][data["edge_index"][0]]+data["radius_vdw"][data["edge_index"][1]])
        # compute edge lengths and vectors (normalized)
        vectors = data["positions"][data["edge_index"][1]] - data["positions"][data["edge_index"][0]] + data["shifts"]  # [n_edges, 3]
        edge_lengths = torch.linalg.norm(vectors, dim=-1, keepdim=False)  # [n_edges]
        # compute the edge energy
        scaled_lengths2 = (kAlpha[data["batch"][data["edge_index"][0]]]*edge_lengths)**2
        energy_edge = -1.0\
            *data[self.key_output_node][data["edge_index"][0]]\
            *data[self.key_output_node][data["edge_index"][1]]\
            *torch.exp(-1.0*scaled_lengths2)\
            *(1.0+scaled_lengths2*(1.0+0.5*scaled_lengths2))/(edge_lengths**6+rvdw**6)
        #energy_edge = 1.0\
        #    *data[self.key_output_node][data["edge_index"][0]]\
        #    *data[self.key_output_node][data["edge_index"][1]]\
        #    *((1.0-torch.exp(-1.0*scaled_lengths2)*(1.0+scaled_lengths2*(1.0+0.5*scaled_lengths2)))/edge_lengths**6\
        #    -1.0/(edge_lengths**6+rvdw**6))
        # compute the node energy
        n_nodes = data["positions"].shape[0]
        energy_node = 0.5*scatter_sum(
            src=energy_edge, 
            index=data["edge_index"][1], 
            dim=0, 
            dim_size=n_nodes
        )*self.weight
        # compute the structure energy
        er = scatter_sum(
            src = energy_node,
            index = data["batch"],
            dim = 0
        )
        #print("er = ",er)

        # == compute the energy - kspace ==
        #print("computing the energy - kspace term")
        results = []
        unique_batches = torch.unique(data["batch"])
        for i in unique_batches:
            mask = data["batch"] == i  # Create a mask for the i-th configuration
            kvecs=(\
                cellK[i,0,:].unsqueeze(-1)*kpoints[:,0]+\
                cellK[i,1,:].unsqueeze(-1)*kpoints[:,1]+\
                cellK[i,2,:].unsqueeze(-1)*kpoints[:,2]\
            ) # [3,nkvec]
            knorms=torch.linalg.norm(kvecs,dim=0) # [nkvec]
            b=0.5*knorms/kAlpha[i]
            kamps=(knorms*np.sqrt(np.pi))**3/(24.0*vol[i])*(\
                np.sqrt(np.pi)*torch.erfc(b)+\
                torch.exp(-1.0*b*b)*(1.0/(2.0*b*b*b)-1.0/b)\
            ) # [nkvec]
            rdotk = torch.matmul(data["positions"][mask],kvecs) # [numnodes,nkvec]
            qrdotk = \
                torch.matmul(out_node[mask],torch.cos(rdotk))**2+\
                torch.matmul(out_node[mask],torch.sin(rdotk))**2 # [nkvec]
            results.append(-1.0*torch.matmul(kamps,qrdotk)) #[]
        ek = torch.stack(results, dim=0)*self.weight
        #print("ek = ",ek)

        # == compute total energy ==
        data[self.key_energy]=er+ek+ec
        
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
        data[self.key_forces] = forces*self.weight
        if self.key_virials is not None:
            data[self.key_virials] = virials*self.weight
        if self.key_stress is not None:
            data[self.key_stress] = stress*self.weight

        # == return ==
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
            # keys - input/output
            f"key_input = {self.key_input}\n"
            f"key_output_reduce = {self.key_output_reduce}\n"
            f"key_output_node = {self.key_output_node}\n"
            # keys - energy/force
            f"key_energy = {self.key_energy}\n"
            f"key_forces = {self.key_forces}\n"
            f"key_virials = {self.key_virials}\n"
            f"key_stress = {self.key_stress}\n"
            # kspace
            f"rc = {self.rc}\n"
            f"prec = {self.prec}\n"
            # elements
            f"radii = {self.radii}\n"
            # neural network
            f"n_in = {self.n_in}\n"
            f"n_out = {self.n_out}\n"
            f"n_hidden = {self.n_hidden}\n"
            f"activation = {self.activation}\n"
            f"skip = {self.skip}\n"
            f"linout = {self.linout}\n"
            f"weight = {self.weight}\n"
            # neural nets
            f"{self.outnet}\n"
            f"{self.linear_nn}\n"
            f"**********************************************"
        )