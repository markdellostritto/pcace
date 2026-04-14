import torch
from enum import Enum
from typing import Dict, Union, Callable, Optional

class NormT(Enum):
    NONE = 1
    LINEAR = 2
    SQRT = 3

"""
    Defines mappings to a loss function and weight for training
"""
class LossMap(torch.nn.Module):
    """
        target_name: Name of target in training batch.
        name: name of the loss object
        loss_fn: function to compute the loss
        loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
            This can be a float or a callable that takes in the loss_weight_args
            For example, if we want the loss weight to be dependent on the epoch number
            if training == True and a default value of 1.0 otherwise,
            loss_weight can be, e.g., lambda training, epoch: 1.0 if not training else epoch / 100
    """
    def __init__(
        self,
        target_name: str,
        predict_name: Optional[str] = None,
        name: Optional[str] = None,
        loss_fn: Optional[torch.nn.Module] = None,
        loss_weight: Union[float, Callable] = 1.0, # Union[float, Callable] means that the type can be either float or callable
        normT: NormT = NormT.NONE,
    ):
        super().__init__()
        self.target_name = target_name
        self.predict_name = predict_name or target_name
        self.name = name or target_name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.normT = normT

    def forward(self, 
        pred: Dict[str, torch.Tensor], 
        target: Optional[Dict[str, torch.Tensor]] = None,
        loss_args: Optional[Dict[str, torch.Tensor]] = None
    ):
        # return nothing if no weight or function is defined
        if self.loss_weight == 0 or self.loss_fn is None: return 0.0
        # set the loss weight if it is a function
        if isinstance(self.loss_weight, Callable):
            if loss_args is None: loss_weight = self.loss_weight()
            else: loss_weight = self.loss_weight(**loss_args)
        else: 
            loss_weight = self.loss_weight
        # collect the predicted tensor
        pred_tensor = pred[self.predict_name]
        if pred_tensor.shape != target[self.target_name].shape:
            pred_tensor = pred_tensor.reshape(target[self.target_name].shape)
        # collect the target tensor
        if target is not None:
            target_tensor = target[self.target_name]
        elif self.predict_name != self.target_name:
            target_tensor = pred[self.target_name]
        else:
            raise ValueError("Target is None and predict_name is not equal to target_name")
        # compute the weighted loss
        nAtoms = torch.bincount(target['batch'])
        if(nAtoms.shape == target_tensor.shape):
            # loss - energy
            match self.normT:
                case NormT.NONE: loss = loss_weight * self.loss_fn(pred_tensor, target_tensor)
                case NormT.LINEAR: loss = loss_weight * self.loss_fn(pred_tensor/nAtoms, target_tensor/nAtoms)
                case NormT.SQRT: loss = loss_weight * self.loss_fn(pred_tensor/torch.sqrt(nAtoms), target_tensor/torch.sqrt(nAtoms))
                case _: raise ValueError('Invalid normalization method.')
        else: 
            # loss - force
            nAtomsV=nAtoms[target['batch']].unsqueeze(-1).expand(-1,3).clone()
            match self.normT:
                case NormT.NONE: loss = loss_weight * self.loss_fn(pred_tensor, target_tensor)
                case NormT.LINEAR: loss = loss_weight * self.loss_fn(pred_tensor/nAtomsV, target_tensor/nAtomsV)
                case NormT.SQRT: loss = loss_weight * self.loss_fn(pred_tensor/torch.sqrt(nAtomsV), target_tensor/torch.sqrt(nAtomsV))
                case _: raise ValueError('Invalid normalization method.')
        # return the loss
        return loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name}, loss_fn={self.loss_fn}, loss_weight={self.loss_weight})"
            )

