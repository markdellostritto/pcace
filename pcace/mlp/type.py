import torch
import torch.nn as nn
from typing import Sequence

__all__ = [
    'NodeEncoder',
    'NodeEmbedder',
    'EdgeEncoder',
]

"""
    NodeEncoder class
    @member zs - table of all possible atomic numbers
    This class uses a table of possible atomic numbers to create
    a one-hot encoding of atomic numbers.
"""
class NodeEncoder(nn.Module):
    #==== initialization ====
    def __init__(self, zlist: Sequence[int]):
        super().__init__()
        self.nz = len(zlist)
        self.register_buffer("zmap", 
            torch.tensor(
                [zlist.index(z) if z in zlist else -1 for z in range(max(zlist) + 1)], 
                dtype=torch.int64
            )
        )

    #==== calculation ====
    def forward(self, zz) -> torch.Tensor:
        # **** convert atomic numbers to indices ****
        indices = self.zmap[zz]
        if (indices < 0).any(): raise ValueError(f"Invalid atomic numbers: {zz[indices < 0]}")
        # **** generate one-hot encoding ****
        shape=indices.shape+(self.nz,) # shape([len(zz),nz])
        onehot=torch.zeros(shape,device=indices.device) # init with zeros
        onehot.scatter_(dim=-1, index=indices.unsqueeze(-1), value=1) # fill index with one
        # **** return one-hot encoding ****
        return onehot
    
    #==== representation ====
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(nz={self.nz})"
        )

"""
    NodeEmbedder class
    @member embedding_weights - the weights used for embedding the node attributes
    This class embeds node attributes, with dimension node_dim, into an embedding, 
    with dimension embedding_dimension.  The weights are normally distributed initially,
    with the possibility of training.
"""
class NodeEmbedder(nn.Module):
    #==== initialization ====
    """
        dim_node - the intial dimension of the nodes (e.g. the number of elements)
        dim_embed - the dimension into which we will embed the node dimension 
        trainable - whether the embedding is trainable
        random_seed - random_seed for initialization
    """
    def __init__(self, dim_node:int, dim_embed:int, trainable=True, random_seed=42):
        super().__init__()
        # resize
        weights = torch.Tensor(dim_node, dim_embed)
        # initialize
        if random_seed is not None: torch.manual_seed(random_seed)
        nn.init.xavier_uniform_(weights)
        # set trainable
        if trainable: self.weights = nn.Parameter(weights,requires_grad=True)
        else: self.register_buffer("weights", weights)

    #==== calculation ====
    def forward(self, node_data: torch.Tensor) -> torch.Tensor:
        return torch.mm(node_data, self.weights) # mm - no broadcasting

    #==== representation ====
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(dim_node={self.weights.shape[0]}, dim_embed={self.weights.shape[1]})"
        )

"""
    EdgeEncoder class
"""
class EdgeEncoder(nn.Module):
    #==== initialization ====
    def __init__(self, directed=True):
        super().__init__()
        self.directed = directed

    #==== calculation ====
    """
        edge_index - list of edges connecting all nodes
        node_type_s - sender node types (n_nodes, dim_node)
        node_type_r - receiver node types (n_nodes, dim_node)
    """
    def forward(self,     
        edge_index: torch.Tensor,  # [2, n_edges]
        node_type_s: torch.Tensor,  # [n_nodes, dim_node]
        node_type_r: torch.Tensor=None,  # [n_nodes, dim_node]
    ) -> torch.Tensor:
        # **** set the receiver type if not provided ****
        if node_type_r is None: node_type_r = node_type_s
        # **** stack all sender/reciever node types in the edge list ****
        node_s = node_type_s[edge_index[0]] # [nedges, dim_node]
        node_r = node_type_r[edge_index[1]] # [nedges, dim_node]
        encoded_edges = torch.einsum('ki,kj->kij', node_s, node_r).flatten(start_dim=1)
        # **** return encoded edges ****
        return encoded_edges # [n_edges, dim_node*dim_node]
    
    #==== representation ====
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(directed={self.directed})"
        )

