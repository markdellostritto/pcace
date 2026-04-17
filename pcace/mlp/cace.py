#*************************************************************************
# Import Statements
#*************************************************************************

import torch
from torch import nn
from typing import Sequence, Optional, List, Dict, Any, Callable

from .ann import ANN
from .type import (
    NodeEncoder, 
    NodeEmbedder, 
    EdgeEncoder,
)
from ..basis import (
    AngularProduct
)
from ..tools import (
    scatter_sum
)
from .force import (
    get_outputs,
    get_symmetric_displacement
)

#*************************************************************************
# CACE Representation
#*************************************************************************

class CACE(nn.Module):
    # ==== initialization ====
    def __init__(
        self,
        # atomic numbers
        z_list: Sequence[int],
        # body order
        order: int,
        # basis
        cutoff: nn.Module,
        radial: nn.Module,
        angular: nn.Module,
        # node/edge encoding/embedding
        dim_node_embed: int,
        # radial embedding
        dim_radial_embed: int,
        # neural network
        n_hidden: Optional[Sequence[int]] = None,
        activation: Callable = torch.nn.SiLU(),
        skip: bool = False,
        # forces
        calc_forces = True,
        calc_virials = True,
        calc_stress = True,
        # message passing
        num_message_passing: int = 0,
        type_message_passing: List[str] = ["M", "Ar", "Bchi"],
        args_message_passing: Dict[str, Any] = {"M": {}, "Ar": {}, "Bchi": {}},
        avg_num_neighbors: float = 10.0,
        keep_node_features_A: bool = False,
        # device
        device: torch.device = torch.device("cpu"),
    ):
        # == init ==
        super().__init__()
        
        # == set constants and flags ==
        self.mp_norm_factor = 1.0/(avg_num_neighbors)**0.5 # normalization factor for message passing
        self.keep_node_features_A = keep_node_features_A
        self.calc_forces = calc_forces
        self.calc_virials = calc_virials
        self.calc_stress = calc_stress
        
        # == atomic numbers ==
        self.z_list = z_list # list of all possible elements
        self.nz = len(z_list) # number of possible elements
        
        # == body order ==
        self.order = order
        
        # == node encoding ==
        self.node_encoder = NodeEncoder(self.z_list)
        
        # == node embedding ==
        self.dim_node_embed = dim_node_embed
        self.node_embedder_send = NodeEmbedder(
            dim_node=self.nz, 
            dim_embed=self.dim_node_embed, 
            random_seed=42
        )
        self.node_embedder_recv = NodeEmbedder(
            dim_node=self.nz, 
            dim_embed=self.dim_node_embed, 
            random_seed=42
        )
        
        # == edge encoding ==
        self.edge_encoder = EdgeEncoder(directed=True)
        self.dim_edge_encode = self.dim_node_embed**2

        # == set cutoff ==
        self.cutoff = cutoff

        # == set radial basis ==
        self.radial = radial
        self.dim_radial = self.radial.nr
        self.dim_radial_embed = dim_radial_embed
        
        # == set angular basis ==
        self.angular = angular
        self.ang_lim=[(angular.beg(l),angular.end(l)) for l in range(0,angular.l_max+1)]
        
        # == set radial transform ==
        self.rt_weights=nn.ParameterList([
            nn.Parameter(torch.rand([self.dim_radial, self.dim_radial_embed, self.dim_edge_encode]),requires_grad=True) 
            for l in range(0,angular.l_max+1)
        ])
        
        # == set angular product ==
        self.ang_prod=AngularProduct(order,angular.l_max)

        # == message passing ==

        # == set the input size ==
        self.n_input = self.dim_radial_embed*self.ang_prod.size*self.dim_edge_encode

        # == build the hamiltonian ==
        self.ann = ANN(
            n_in = self.n_input,
            n_out = 1,
            n_hidden = n_hidden,
            activation = activation,
            skip = skip,
        )

        # == device ==
        self.device = device
    
    # ==== calculation ====
    def forward(
        self, 
        data: Dict[str, torch.Tensor],
        training = True,
    ):
        #print("Cace::forward")
        data["positions"].requires_grad_(True)
        
        # == get the network data ==
        #print("get network data")
        n_nodes = data['positions'].shape[0]
        if data["batch"] == None: batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=self.device)
        else: batch_now = data["batch"]
        try:
            data["num_graphs"]=data["ptr"].numel()-1
        except:
            data["num_graphs"]=1
        #print("n_nodes = ",n_nodes)
        
        # == get symmetric displacement ==
        #print("get symmetric displacement")
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
            
        # == node encoding ==
        node_encoding = self.node_encoder(data['atomic_numbers'])
        #print("node_encoding = ",node_encoding)

        # == node embedding ==
        node_embedding_send = self.node_embedder_send(node_encoding)
        node_embedding_recv = self.node_embedder_recv(node_encoding)
        #print("node_embedding_send = ",node_embedding_send)
        #print("node_embedding_recv = ",node_embedding_recv)

        # == edge encoding == 
        # edge_encoding : [n_edges, dim_edge_encode]
        edge_encoding = self.edge_encoder(
            edge_index=data["edge_index"],
            node_type_s=node_embedding_send,
            node_type_r=node_embedding_recv
        )
        #print("edge_encoding = ",edge_encoding)

        # == compute edge lengths and vectors (normalized) ==
        vectors = data["positions"][data["edge_index"][1]] - data["positions"][data["edge_index"][0]] + data["shifts"]  # [n_edges, 3]
        edge_lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
        edge_vectors = vectors / (edge_lengths + 1e-12)
        
        # == compute angular and radial terms ==
        radial_component = self.radial(edge_lengths) 
        radial_cutoff = self.cutoff(edge_lengths)
        angular_component = self.angular(edge_vectors)
        #print("radial_component = ",radial_component)
        #print("radial_cutoff = ",radial_cutoff)
        #print("angular_component = ",angular_component)
        
        # == combine to form edge attributes == 
        # edge_attri : [n_edges, dim_radial, angular_dim, dim_edge_encode]
        edge_attri = torch.einsum('ni,nj,nk->nijk',
            radial_component * radial_cutoff,
            angular_component,
            edge_encoding
        )
        #edge_attri = \
        #    (radial_component * radial_cutoff).unsqueeze(2).unsqueeze(3)*\
        #    angular_component.unsqueeze(1).unsqueeze(3) *\
        #    edge_encoding.unsqueeze(1).unsqueeze(2)
        #print("edge_attri = ",edge_attri.size())

        # == sum over edge features to each node ==
        # node_A : [n_nodes, dim_radial, angular_dim, dim_edge_encode]
        node_A = scatter_sum(
            src=edge_attri, 
            index=data["edge_index"][1], 
            dim=0, 
            dim_size=n_nodes
        )
        #print("node_A = ",node_A.size())

        # == mix the different radial components ==
        node_T = torch.zeros((n_nodes, self.dim_radial_embed, self.angular.size, self.dim_edge_encode),device=self.device)
        for l, weight in enumerate(self.rt_weights):
            # set the beg and end to get the angular components with the same total angular momentum
            #print("angmom_lim[",l,"] = ",self.ang_lim[l])
            i_beg = self.ang_lim[l][0]
            i_end = self.ang_lim[l][1]
            lgroup = torch.arange(i_beg, i_end)
            # Gather all angular dimensions for the current group
            group_x = node_A[:, :, lgroup, :]  # Shape: [n_nodes, radial_dim, len(lgroup), embedding_dim]
            # Apply the transformation for the entire group at once
            transformed_group = torch.einsum('ijkh,jmh->imkh', group_x, weight)
            # Assign to the output tensor for each angular dimension
            node_T[:, :, lgroup, :] = transformed_group
        #print("node_T = ",node_T.size())
            
        # == symmetrize the basis ==
        node_S = torch.zeros((n_nodes, self.dim_radial_embed, self.ang_prod.size, self.dim_edge_encode),device=self.device)
        #print("node_S = ",node_S.size())
        #print("n_nodes = ",n_nodes)
        #print("dim_radial_embed = ",self.dim_radial_embed)
        #print("ang_prod = ",self.ang_prod.size)
        #print("dim_edge_encode = ",self.dim_edge_encode)
        #print("ang_prod.lprod = ",self.ang_prod.lprod)
        #print("ang_prod.p_size = ",self.ang_prod.p_size)
        #print("ang_prod.offset = ",self.ang_prod.offset)
        if(self.order>=1):
            #print("order >= 1")
            #print("beg = ",self.ang_prod.beg(1))
            #print("end = ",self.ang_prod.end(1))
            node_S[:, :, 0, :] = node_T[:, :, 0, :]
        if(self.order>=2):
            #print("order >= 2")
            #print("beg = ",self.ang_prod.beg(2))
            #print("end = ",self.ang_prod.end(2))
            for n in range(self.ang_prod.beg(2),self.ang_prod.end(2)):
                lvec = self.ang_prod.lprod[n]
                lim0 = torch.arange(self.angular.beg(lvec[0]),self.angular.end(lvec[0]))
                node_S[:, :, n, :] = torch.einsum("abic,abic->abc",
                    node_T[:, :, lim0, :],
                    node_T[:, :, lim0, :]
                )
        if(self.order>=3):
            #print("order >= 3")
            #print("beg = ",self.ang_prod.beg(3))
            #print("end = ",self.ang_prod.end(3))
            for n in range(self.ang_prod.beg(3),self.ang_prod.end(3)):
                lvec = self.ang_prod.lprod[n]
                lim0 = torch.arange(self.angular.beg(lvec[0]),self.angular.end(lvec[0]))
                lim1 = torch.arange(self.angular.beg(lvec[1]),self.angular.end(lvec[1]))
                lim0p1 = torch.arange(self.angular.beg(lvec[0]+lvec[1]),self.angular.end(lvec[0]+lvec[1]))
                vshape0p1=node_T[:, :, lim0p1, :].shape
                node_S[:, :, n, :] = torch.einsum("abic,abijc,abjc->abc",
                    node_T[:, :, lim0, :],
                    node_T[:, :, lim0p1, :].reshape(vshape0p1[0],vshape0p1[1],self.angular.l_size[lvec[0]],self.angular.l_size[lvec[1]],vshape0p1[3]),
                    node_T[:, :, lim1, :]
                )
        if(self.order>=4):
            #print("order >= 4")
            #print("beg = ",self.ang_prod.beg(3))
            #print("end = ",self.ang_prod.end(3))
            for n in range(self.ang_prod.beg(4),self.ang_prod.end(4)):
                lvec = self.ang_prod.lprod[n]
                lim0 = torch.arange(self.angular.beg(lvec[0]),self.angular.end(lvec[0]))
                lim1 = torch.arange(self.angular.beg(lvec[1]),self.angular.end(lvec[1]))
                lim2 = torch.arange(self.angular.beg(lvec[2]),self.angular.end(lvec[2]))
                lim0p1 = torch.arange(self.angular.beg(lvec[0]+lvec[1]),self.angular.end(lvec[0]+lvec[1]))
                lim1p2 = torch.arange(self.angular.beg(lvec[1]+lvec[2]),self.angular.end(lvec[1]+lvec[2]))
                vshape0p1=node_T[:, :, lim0p1, :].shape
                vshape1p2=node_T[:, :, lim1p2, :].shape
                node_S[:, :, n, :] = torch.einsum("abic,abijc,abjkc,abkc->abc",
                    node_T[:, :, lim0, :],
                    node_T[:, :, lim0p1, :].reshape(vshape0p1[0],vshape0p1[1],self.angular.l_size[lvec[0]],self.angular.l_size[lvec[1]],vshape0p1[3]),
                    node_T[:, :, lim1p2, :].reshape(vshape1p2[0],vshape1p2[1],self.angular.l_size[lvec[1]],self.angular.l_size[lvec[2]],vshape1p2[3]),
                    node_T[:, :, lim2, :]
                )
        data["node_feats"] = node_S
        #print("sym_node_attr = ",node_S.size())
        #print("n_input = ",self.n_input)
        
        # == message passing ==
        # NOTE: TODO
        
        # == compute the energy ==
        #print("batch indices = ",data["batch"])
        self.ann.forward(data)
        total_energy=scatter_sum(
            src=data["node_energy"],
            index=data["batch"],
            dim=0
        )
        total_energy=torch.squeeze(total_energy,-1)
        #print("node_energy = ",data["node_energy"])
        #print("total_energy = ",total_energy)
        #natoms = torch.bincount(data['batch'])
        #natomsv = torch.tensor([natoms[b] for b in data["batch"]]).unsqueeze(-1).expand(-1,3)

        # == compute forces ==
        #print("computing forces")
        forces, virials, stress = get_outputs(
            energy = total_energy,
            positions = data['positions'],
            displacement = data.get('displacement', None),
            cell = data.get('cell', None),
            training=training,
            compute_force = self.calc_forces,
            compute_virials = self.calc_virials,
            compute_stress = self.calc_stress
        )
        #print(data["ptr"].numel())
        #print("forces = ",forces)
        #print("forces_norm = ",forces/countsv.view(-1,1))
        #print("stress = ",stress)
        #print("virials = ",virials)
        
        # == build the output ==
        try: displacement = data["displacement"]
        except: displacement = None
        output = {
            "positions": data["positions"],
            "energy": total_energy,
            "cell": data["cell"],
            "displacement": displacement,
            "batch": batch_now,
            "node_feats": node_S,
            "forces": forces,
            "virials": virials,
            "stress": stress,
        }
        
        # == return the output ==
        return output
    
    # ==== output ====
    def __repr__(self):
        return (
            f"\n**********************************************\n"
            f"{self.__class__.__name__}\n"
            f"z_list = {self.z_list}\n"
            f"order  = {self.order}\n"
            f"cutoff = {self.cutoff}\n"
            f"radial  = {self.radial}\n"
            f"angular = {self.angular}\n"
            f"product = {self.ang_prod}\n"
            f"dim_node_embed   = {self.dim_node_embed}\n"
            f"dim_edge_encode  = {self.dim_edge_encode}\n"
            f"dim_radial_embed = {self.dim_radial_embed}\n"
            f"node_encoder = {self.node_encoder}\n"
            f"node_embedder_send = {self.node_embedder_send}\n"
            f"node_embedder_recv = {self.node_embedder_recv}\n"
            f"calc_forces  = {self.calc_forces}\n"
            f"calc_virials = {self.calc_virials}\n"
            f"calc_stress  = {self.calc_stress}\n"
            f"device = {self.device}\n"
            f"{self.rt_weights}\n"
            f"{self.ann}\n"
            f"**********************************************"
        )
