import torch
from torch import nn
from typing import Callable, Optional, Sequence

__all__ = ["MLP"]

"""
    Multilayer Perceptron
"""
class MLP(nn.Module):
    # ==== initialization ====
    def __init__(
        self,
        n_in: int, n_out: int,
        n_hidden: Optional[Sequence[int]] = None,
        activation: Callable = nn.SiLU(),
        bias: bool = True,
    ):
        super().__init__()
        # ---- set the neural network architecture ----
        if n_hidden is not None:
            n_neurons = [n_in] + n_hidden + [n_out]
        else: 
            n_neurons = [n_in] + [n_out]
        # ---- create the layers ----
        self.layers = nn.Sequential()
        for i in range(len(n_neurons)-2):
            self.layers.append(nn.Linear(n_neurons[i],n_neurons[i+1],bias))
            if(activation is not None): self.layers.append(activation)
        self.layers.append(nn.Linear(n_neurons[-2],n_neurons[-1],bias))
        
    # ==== calculation ====
    def forward(self, input: torch.Tensor):
        return self.layers(input)
