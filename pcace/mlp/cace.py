#*************************************************************************************************
# Import Statements
#*************************************************************************************************

import torch
from typing import Sequence, List, Dict, Any

from .type import (
    NodeEncoder, 
    NodeEmbedder, 
    EdgeEncoder,
)
from ..tools import (
    scatter_sum
)

#*************************************************************************************************
# CACE Representation
#*************************************************************************************************

class CACE(torch.nn.Module):
    # ==== initialization ====
    def __init__(
        self,
        # atomic numbers
        z_list: Sequence[int],
        # basis
        cutoff: torch.nn.Module,
        radial: torch.nn.Module,
        angular: torch.nn.Module,
        angprod: torch.nn.Module,
        # node/edge encoding/embedding
        dim_node_embed: int,
        # radial embedding
        dim_radial_embed: int,
        # message passing
        num_message_passing: int = 0,
        type_message_passing: List[str] = ["M", "Ar", "Bchi"],
        args_message_passing: Dict[str, Any] = {"M": {}, "Ar": {}, "Bchi": {}},
        avg_num_neighbors: float = 10.0,
        keep_node_features_A: bool = False,
    ):
        # == init ==
        super().__init__()

        # == set constants and flags ==
        #self.mp_norm_factor = 1.0/(avg_num_neighbors)**0.5 # normalization factor for message passing
        #self.keep_node_features_A = keep_node_features_A
        
        # == atomic numbers ==
        self.z_list = z_list # list of all possible elements
        self.nz = len(z_list) # number of possible elements
        
        # == node encoding ==
        self.node_encoder = NodeEncoder(self.z_list)
        
        # == node embedding ==
        self.dim_node_embed = dim_node_embed
        self.node_embedder_send = NodeEmbedder(
            dim_node    = self.nz, 
            dim_embed   = self.dim_node_embed, 
            random_seed = 42
        )
        self.node_embedder_recv = NodeEmbedder(
            dim_node    = self.nz, 
            dim_embed   = self.dim_node_embed, 
            random_seed = 42
        )
        
        # == edge encoding ==
        self.edge_encoder = EdgeEncoder(directed=True)
        #self.dim_edge_encode = self.dim_node_embed**2
        self.register_buffer(
            "dim_edge_encode", torch.tensor(self.dim_node_embed**2, dtype=torch.int)
        )

        # == set cutoff ==
        self.cutoff = cutoff

        # == set radial basis ==
        self.radial = radial
        self.dim_radial = self.radial.nr
        #self.dim_radial_embed = dim_radial_embed
        self.register_buffer(
            "dim_radial_embed", torch.tensor(dim_radial_embed, dtype=torch.int)
        )
        
        # == set angular basis ==
        self.angular = angular
        
        # == set radial transform ==
        self.rt_weights = torch.nn.ParameterList([
            torch.nn.Parameter(torch.rand([
                self.dim_radial, self.dim_radial_embed, self.dim_edge_encode
            ]),requires_grad=True) 
            for l in range(0,angular.l_max+1)
        ])
        
        # == set angular product ==
        self.angprod = angprod

        # == message passing ==

        # == set the input size ==
        self.n_input = self.dim_radial_embed*self.angprod.size*self.dim_edge_encode

        
    # ==== calculation ====
    def forward(
        self, 
        data: Dict[str, torch.Tensor],
    ):
        # == get the device ==
        device = data["batch"].device
        #print("device = ",device)

        # == get the network data ==
        n_nodes = data["atomic_numbers"].shape[0]
        #print("n_nodes = ",n_nodes)
        
        # == node encoding ==
        node_encoding = self.node_encoder(data["atomic_numbers"]) # [n_nodes, n_elements]
        #print("node_encoding = ",node_encoding)
        
        # == node embedding ==
        node_embedding_send = self.node_embedder_send(node_encoding) # [n_nodes, dim_node]
        node_embedding_recv = self.node_embedder_recv(node_encoding) # [n_nodes, dim_node]
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
        # When running MD simulations (i.e. LAMMPS) the vectors already exist and 
        # should not be computed from the positions.  If the vectors are not already
        # stored, then they must be computed from the positions and edge indices
        if "vectors" not in data: 
            data["vectors"] = data["positions"][data["edge_index"][1]] \
                - data["positions"][data["edge_index"][0]] \
                + data["shifts"]  # [n_edges, 3]
        edge_lengths = torch.linalg.norm(data["vectors"], dim=-1, keepdim=True)  # [n_edges, 1]
        edge_vectors = data["vectors"] / (edge_lengths + 1e-16) # [n_edges, 3]
        #print("nedges = ",edge_lengths.size)
        
        # == compute angular and radial terms ==
        radial_component = self.radial(edge_lengths) # [n_edges, dim_radial]
        radial_cutoff = self.cutoff(edge_lengths) # [n_edges, 1]
        angular_component = self.angular(edge_vectors) # [n_edges, dim_angular]
        #print("radial_component = ",radial_component)
        #print("radial_cutoff = ",radial_cutoff)
        #print("angular_component = ",angular_component)
        
        # == combine to form edge attributes == 
        # edge_attri : [n_edges, dim_radial, dim_angular, dim_edge_encode]
        # einsum : easier to read but slower
        #edge_attri = torch.einsum('ni,nj,nk->nijk',
        #    radial_component * radial_cutoff, # [n_edges, dim_radial]
        #    angular_component, # [n_edges, dim_angular]
        #    edge_encoding # [n_edges, dim_edge_encode]
        #)
        # unsqueeze : harder to read but faster
        edge_attri = \
            (radial_component * radial_cutoff).unsqueeze(2).unsqueeze(3)*\
            angular_component.unsqueeze(1).unsqueeze(3)*\
            edge_encoding.unsqueeze(1).unsqueeze(2)
        #print("edge_attri = ",edge_attri.size())
        
        # == sum over edge features to each node ==
        # node_A : [n_nodes, dim_radial, dim_angular, dim_edge_encode]
        node_A = scatter_sum(
            src=edge_attri,
            index=data["edge_index"][1],
            dim=0,
            dim_size=n_nodes
        )
        #print("node_A = ",node_A.size())
        
        # == mix the different radial components ==
        # node_T : [n_nodes, dim_radial_embed, dim_angular, dim_edge_encode]
        node_T = torch.zeros((
            n_nodes,
            self.dim_radial_embed,
            self.angular.size,
            self.dim_edge_encode),
        device=device)
        for l, weight in enumerate(self.rt_weights):
            # set the beg and end to get the angular components with the same total angular momentum
            lgroup = torch.arange(self.angular.lvec_lim[l][0], self.angular.lvec_lim[l][1],device=device)
            # Apply the transformation for all angular dims in the entire group at once
            node_T[:, :, lgroup, :] = torch.einsum('ijkh,jmh->imkh', node_A[:, :, lgroup, :], weight)
        #print("node_T = ",node_T.size())
            
        # == symmetrize the basis ==
        # node_S : [n_nodes, dim_radial_embed, dim_ang_prod, dim_edge_encode]
        dim_ang_prod = self.angprod.size
        node_S = torch.zeros((
            n_nodes, 
            self.dim_radial_embed, 
            dim_ang_prod, 
            self.dim_edge_encode),
        device=device)

        # == symmetrize ==
        if(self.angprod.o_max>=1):
            #print(f"o_max >= 1 ({self.angprod.beg(1)},{self.angprod.end(1)})")
            node_S[:, :, 0, :] = node_T[:, :, 0, :]
        if(self.angprod.o_max>=2):
            #print(f"o_max >= 2 ({self.angprod.beg(2)},{self.angprod.end(2)})")
            for n in range(self.angprod.beg(2),self.angprod.end(2)):
                lvec = self.angprod.lprod[n]
                lvec_lim = self.angular.lvec_lim
                lim0 = torch.arange(lvec_lim[lvec[0]][0],lvec_lim[lvec[0]][1],device=device)
                node_S[:, :, n, :] = torch.einsum("abic,abic->abc",
                    node_T[:, :, lim0, :],
                    node_T[:, :, lim0, :]
                )
        if(self.angprod.o_max>=3):
            #print(f"o_max >= 3 ({self.angprod.beg(3)},{self.angprod.end(3)})")
            for n in range(self.angprod.beg(3),self.angprod.end(3)):
                lvec = self.angprod.lprod[n]
                lvec_lim = self.angular.lvec_lim
                lim0 = torch.arange(lvec_lim[lvec[0]][0],lvec_lim[lvec[0]][1],device=device)
                lim0p1 = torch.arange(lvec_lim[lvec[0]+lvec[1]][0],lvec_lim[lvec[0]+lvec[1]][1],device=device)
                lim1 = torch.arange(lvec_lim[lvec[1]][0],lvec_lim[lvec[1]][1],device=device)
                vshape0p1=node_T[:, :, lim0p1, :].shape
                node_S[:, :, n, :] = torch.einsum("abic,abijc,abjc->abc",
                    node_T[:, :, lim0, :],
                    node_T[:, :, lim0p1, :].reshape(vshape0p1[0],vshape0p1[1],self.angular.l_size[lvec[0]],self.angular.l_size[lvec[1]],vshape0p1[3]),
                    node_T[:, :, lim1, :]
                )
        if(self.angprod.o_max>=4):
            #print(f"o_max >= 4 ({self.angprod.beg(4)},{self.angprod.end(4)})")
            for n in range(self.angprod.beg(4),self.angprod.end(4)):
                lvec = self.angprod.lprod[n]
                lvec_lim = self.angular.lvec_lim
                lim0 = torch.arange(lvec_lim[lvec[0]][0],lvec_lim[lvec[0]][1],device=device)
                lim0p1 = torch.arange(lvec_lim[lvec[0]+lvec[1]][0],lvec_lim[lvec[0]+lvec[1]][1],device=device)
                lim1p2 = torch.arange(lvec_lim[lvec[1]+lvec[2]][0],lvec_lim[lvec[1]+lvec[2]][1],device=device)
                lim2 = torch.arange(lvec_lim[lvec[2]][0],lvec_lim[lvec[2]][1],device=device)
                vshape0p1=node_T[:, :, lim0p1, :].shape
                vshape1p2=node_T[:, :, lim1p2, :].shape
                node_S[:, :, n, :] = torch.einsum("abic,abijc,abjkc,abkc->abc",
                    node_T[:, :, lim0, :],
                    node_T[:, :, lim0p1, :].reshape(vshape0p1[0],vshape0p1[1],self.angular.l_size[lvec[0]],self.angular.l_size[lvec[1]],vshape0p1[3]),
                    node_T[:, :, lim1p2, :].reshape(vshape1p2[0],vshape1p2[1],self.angular.l_size[lvec[1]],self.angular.l_size[lvec[2]],vshape1p2[3]),
                    node_T[:, :, lim2, :]
                )

        # == set the node features ==
        data["node_feats"] = node_S # [n_nodes, dim_radial_embed, dim_ang_prod, dim_edge_encode]
        #print("nod_S.size() = ",node_S.size())
        
        # == message passing ==
        # NOTE: TODO
        
        # == return the features ==
        return data
    
    # ==== output ====
    def __repr__(self):
        return (
            f"\n==============================================\n"
            f"{self.__class__.__name__}\n"
            f"z_list = {self.z_list}\n"
            f"cutoff = {self.cutoff}\n"
            f"radial  = {self.radial}\n"
            f"angular = {self.angular}\n"
            f"product = {self.angprod}\n"
            f"n_input = {self.n_input}\n"
            f"dim_node_embed   = {self.dim_node_embed}\n"
            f"dim_edge_encode  = {self.dim_edge_encode}\n"
            f"dim_radial_embed = {self.dim_radial_embed}\n"
            f"node_encoder = {self.node_encoder}\n"
            f"node_embedder_send = {self.node_embedder_send}\n"
            f"node_embedder_recv = {self.node_embedder_recv}\n"
            f"=============================================="
        )