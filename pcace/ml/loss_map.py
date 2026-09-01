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
        name_target: name of target in training data
        name_oredict: name of target in the predicted data
        name_loss: name of the loss object
        loss_fn: function to compute the loss
        loss_wt: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
            This can be a float or a callable that takes in the loss_weight_args
            For example, if we want the loss weight to be dependent on the epoch number
            if training == True and a default value of 1.0 otherwise,
            loss_wt can be, e.g., lambda training, epoch: 1.0 if not training else epoch / 100
    """
    # ==== initialization ====
    def __init__(
        self,
        # names
        name_target: str,
        name_predict: Optional[str] = None,
        name_loss: Optional[str] = None,
        # loss function
        loss_fn: Optional[torch.nn.Module] = None,
        loss_wt: Union[float, Callable] = 1.0, 
        normT: NormT = NormT.NONE,
    ):
        super().__init__()
        # set the names
        self.name_target = name_target
        self.name_predict = name_predict or name_target
        self.name_loss = name_loss or name_target
        # set the loss function
        self.loss_fn = loss_fn
        self.loss_wt = loss_wt
        self.normT = normT

    # ==== calculation ====
    def forward(self, 
        pred: Dict[str, torch.Tensor], 
        target: Optional[Dict[str, torch.Tensor]] = None,
        loss_args: Optional[Dict[str, torch.Tensor]] = None
    ):
        # return nothing if no weight or function is defined
        if self.loss_wt == 0 or self.loss_fn is None: return 0.0
        # set the loss weight if it is a function
        if isinstance(self.loss_wt, Callable):
            if loss_args is None: loss_wt = self.loss_wt()
            else: loss_wt = self.loss_wt(**loss_args)
        else: 
            loss_wt = self.loss_wt
        # collect the predicted tensor
        pred_tensor = pred[self.name_predict]
        if pred_tensor.shape != target[self.name_target].shape:
            pred_tensor = pred_tensor.reshape(target[self.name_target].shape)
        # collect the target tensor
        if target is not None:
            target_tensor = target[self.name_target]
        elif self.name_predict != self.name_target:
            target_tensor = pred[self.name_target]
        else:
            raise ValueError("Target is None and name_predict is not equal to name_target")
        # compute the weighted loss
        nAtoms = torch.bincount(target['batch'])
        if(nAtoms.shape == target_tensor.shape):
            # loss - energy
            match self.normT:
                case NormT.NONE: 
                    loss = loss_wt * self.loss_fn(pred_tensor, target_tensor)
                case NormT.LINEAR: 
                    loss = loss_wt * self.loss_fn(pred_tensor/nAtoms, target_tensor/nAtoms)
                case NormT.SQRT: 
                    loss = loss_wt * self.loss_fn(pred_tensor/torch.sqrt(nAtoms), target_tensor/torch.sqrt(nAtoms))
                case _: raise ValueError('Invalid normalization method.')
        else: 
            # loss - force
            match self.normT:
                case NormT.NONE: 
                    loss = loss_wt * self.loss_fn(pred_tensor, target_tensor)
                case NormT.LINEAR: 
                    nAtomsV=nAtoms[target['batch']].unsqueeze(-1).expand(-1,3).clone()
                    loss = loss_wt * self.loss_fn(pred_tensor/nAtomsV, target_tensor/nAtomsV)
                case NormT.SQRT: 
                    nAtomsV=nAtoms[target['batch']].unsqueeze(-1).expand(-1,3).clone()
                    loss = loss_wt * self.loss_fn(pred_tensor/torch.sqrt(nAtomsV), target_tensor/torch.sqrt(nAtomsV))
                case _: raise ValueError('Invalid normalization method.')
        # return the loss
        return loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name_loss}, loss_fn={self.loss_fn}, loss_wt={self.loss_wt})"
            )

