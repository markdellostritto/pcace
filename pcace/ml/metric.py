import logging
import torch
from typing import Optional, Dict, List

__all__ = ['Metrics', 'compute_loss_metrics']

"""
    Compute the loss metrics
    current options: mse, mae, rmse, r2
"""
def compute_loss_metrics(metric: str, y_true: torch.Tensor, y_pred: torch.Tensor):
    if metric == 'mse':
        return torch.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return torch.mean(torch.abs(y_true - y_pred))
    elif metric == 'rmse':
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    elif metric == 'r2':
        return 1 - torch.sum((y_true - y_pred) ** 2) / torch.sum((y_true - torch.mean(y_true)) ** 2)
    else:
        raise ValueError('Metric not implemented')

"""
    Defines and calculate  metrics to be logged.
"""
class Metrics(torch.nn.Module):
    #==== initialization ====
    """
        name_target: name of the target in the dataset
        name_predict: name of the prediction in the model output
        name_metric: name of the metrics
        keys_metric: list of metrics to be calculated
        per_atom: whether to calculate the metrics per atom
    """
    def __init__(
        self,
        name_target: str,
        name_predict: Optional[str] = None,
        name_metric: Optional[str] = None,
        keys_metric: List[str] = ["mae", "rmse"],
        per_atom: bool = False,
    ):
        super().__init__()
        self.name_target = name_target
        self.name_predict = name_predict or name_target
        self.name_metric = name_metric or name_target
        self.per_atom = per_atom
        self.keys_metric = keys_metric
        self.logs = {
            "train": {'pred': [], 'target': []},
            "val": {'pred': [], 'target': []},
            "test": {'pred': [], 'target': []},
        }
        
    #==== tensor collection utility ====
    def _collect_tensor(self,
        pred: Dict[str, torch.Tensor],
        target: Optional[Dict[str, torch.Tensor]] = None,
    ):
        pred_tensor = pred[self.name_predict].clone().detach()
        if len(pred_tensor.shape) > 2:
            pred_tensor = pred_tensor.reshape(pred_tensor.shape[0], -1)
        if target is not None:
            target_tensor = target[self.name_target].clone().detach()
        elif self.name_predict != self.name_target:
            target_tensor = pred[self.name_target].clone().detach()
        else:
            raise ValueError("Target is None and name_predict is not equal to name_target")
        if self.per_atom:
            n_atoms = torch.bincount(target['batch']).clone().detach()
            pred_tensor = pred_tensor / n_atoms
            target_tensor = target_tensor / n_atoms
        return pred_tensor, target_tensor
    
    #==== calculation ====
    def forward(self, 
        pred: Dict[str, torch.Tensor],
        target: Optional[Dict[str, torch.Tensor]] = None,
    ):
        pred_tensor, target_tensor = self._collect_tensor(pred, target)
        metrics = {}
        for metric in self.keys_metric:
            metrics[metric] = compute_loss_metrics(metric, target_tensor, pred_tensor)
        return metrics

    #==== metric updater ====
    def update_metrics(self, subset: str, 
        pred: Dict[str, torch.Tensor], 
        target: Optional[Dict[str, torch.Tensor]] = None,
    ):
        pred_tensor, target_tensor = self._collect_tensor(pred, target)
        self.logs[subset]['pred'].append(pred_tensor)
        self.logs[subset]['target'].append(target_tensor)

    def retrieve_metrics(self, subset: str, clear: bool = True, print_log: bool = True):
        # get tensors
        pred_tensor = torch.cat(self.logs[subset]['pred'], dim=0)
        target_tensor = torch.cat(self.logs[subset]['target'], dim=0)
        # check tensors
        assert pred_tensor.shape == target_tensor.shape, f"pred_tensor.shape: {pred_tensor.shape}, target_tensor.shape: {target_tensor.shape}"
        if pred_tensor.shape[0] == 0: raise ValueError("No data in the logs")
        # compute metrics
        metrics = {}
        for metric in self.keys_metric:
            metric_mean = compute_loss_metrics(metric, target_tensor, pred_tensor)
            metrics[metric] = metric_mean
            if print_log:
                print(f'{subset}_{self.name_metric}_{metric}: {metric_mean:.6f}',)
            logging.info(f'{subset}_{self.name_metric}_{metric}: {metric_mean:.6f}',)
        if clear:
            self.clear_metrics(subset)
        # return metrics
        return metrics

    #==== clear metrics ====
    def clear_metrics(self, subset: str):
        self.logs[subset]['pred'] = []
        self.logs[subset]['target'] = []

    #==== output ====
    def __repr__(self):
        return f'{self.__class__.__name__} name: {self.name_metric}, keys_metric: {self.keys_metric}'
