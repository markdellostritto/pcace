#****************************************************
# Import Statements
#****************************************************

import torch
from typing import Dict, Union, Sequence, Callable, Optional
from ..ml import MLP

#****************************************************
# Atomic Neural Network
#****************************************************

"""
    Atomic Neural Network
    Computes properties of a single atom using a neural network
"""
class ANN(torch.nn.Module):
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
        n_in: int = None,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        activation: Callable = torch.nn.SiLU(),
        key_input: Union[str, Sequence[int]] = 'node_feats',
        key_output: str = "node_energy",
        skip: bool = False,
        linout: bool = True,
    ):
        # == init ==
        super().__init__()

        # == set nn data ==
        self.n_out = n_out
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.activation = activation

        # == set keys ==
        self.key_input = key_input
        self.key_output = key_output

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
        y = self.outnet(features)
        if self.skip: y += self.linear_nn(features)

        # == return data ==
        data[self.key_output] = y
        return data

