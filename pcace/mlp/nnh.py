from typing import Dict, Union, Sequence, Callable, Optional
import torch
from ..ml import (
    MLP
)

"""
    Stores and executes the Neural Network Hamiltonian
"""
class NNH(torch.nn.Module):
    """
        n_in: input dimension of representation
        n_out: output dimension of target property (default: 1)
        n_hidden: size of hidden layers.
        output_key: the key under which the result will be stored
    """    
    def __init__(
        self,
        n_in: int = None,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        activation: Callable = torch.nn.SiLU(),
        feature_key: Union[str, Sequence[int]] = 'node_feats',
        output_key: str = "node_energy",
        skip: bool = False
    ):
        super().__init__()
        self.output_key = output_key
        
        self.n_out = n_out
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.activation = activation
        self.feature_key = feature_key

        self.outnet = MLP(
            n_in=self.n_in,
            n_out=self.n_out,
            n_hidden=self.n_hidden,
            activation=self.activation,
        )
        self.skip = skip
        if self.skip:
            self.linear_nn = MLP(
                self.n_in, 
                self.n_out,
                activation=None, 
            ) 
        else: self.linear_nn = None
        
    def forward(self, 
        data: Dict[str, torch.Tensor],
        training: bool = None,
    ) -> Dict[str, torch.Tensor]:
        
        # check if self.feature_key exists, otherwise set default 
        if not hasattr(self, "feature_key") or self.feature_key is None: self.feature_key = "node_feats"

        # check for the features
        if self.feature_key not in data: raise ValueError(f"Feature key {self.feature_key} not found in data dictionary.")
        # extract data
        features = data[self.feature_key]
        # reshape such that each node has its own entry
        features = features.reshape(features.shape[0], -1)
        
        # predict atomwise contributions (one for each node)
        y = self.outnet(features)
        if self.skip: y += self.linear_nn(features)

        # return data
        data[self.output_key] = y
        return data

