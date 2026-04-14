from typing import Optional, Dict, List, Type, Any
import logging
import torch
from torch import nn
import numpy as np
from ..ml import LossMap
from ..ml import Metrics

__all__ = ['TrainingTask']

"""
    The training loop for the neural network model.
"""
class TrainingTask(nn.Module):
    """
        model: the neural network model
        losses: list of losses an optional loss functions
        metrics: list of metrics
        optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
        optimizer_args: dict of optimizer keyword arguments
        scheduler_cls: type of torch learning rate scheduler
        scheduler_args: dict of scheduler keyword arguments
        ema: whether to use exponential moving average
        ema_decay: decay rate of ema
        ema_start: when to start ema
        max_grad_norm: max gradient norm
        warmup_steps: number of warmup steps before reaching the base learning rate
    """
    def __init__(self, 
        model: nn.Module,
        losses: List[LossMap],
        metrics: List[Metrics],
        device: torch.device = torch.device('cpu'),
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        ema: bool = False, 
        ema_decay: float = 0.99,
        ema_start: int = 0,
        max_grad_norm: float = 10,
        warmup_steps: int = 0,                
    ):
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.losses = nn.ModuleList(losses)
        self.metrics = nn.ModuleList(metrics)
        self.optimizer = optimizer_cls(self.parameters(), **optimizer_args)
        self.scheduler = scheduler_cls(self.optimizer, **scheduler_args) if scheduler_cls else None
        self.ema = ema
        self.ema_start = ema_start
        if self.ema:
            # AveragedModel is kinda new in torch so there's a fall back
            try:
                self.ema_model = torch.optim.swa_utils.AveragedModel(self.model, \
                    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay))
            except:
                ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
                    ema_decay * averaged_model_parameter + (1-ema_decay) * model_parameter
                self.ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
        else:
            self.ema_model = None

        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.lr = optimizer_args['lr']
        self.global_step = 0

        #self.grad_enabled = len(self.model.required_derivatives) > 0
        self.grad_enabled = True

    def update_loss(self, losses: List[LossMap]):
        self.losses = nn.ModuleList(losses)

    def update_metrics(self, metrics: List[Metrics]):
        self.metrics = nn.ModuleList(metrics)

    def forward(self, data, training: bool):
        return self.model(data, training=training)

    def loss_fn(self, pred, batch, loss_args: Optional[Dict[str, torch.Tensor]] = None, index: Optional[List[int]] = None):
        loss = 0.0
        if index is not None:
            for i in index:
                loss += self.losses[i](pred, batch, loss_args)
        else:
            for eachloss in self.losses:
                loss += eachloss(pred, batch, loss_args)
        return loss

    def log_metrics(self, subset, pred, batch):
        for metric in self.metrics:
            metric.update_metrics(subset, pred, batch)

    def retrieve_metrics(self, subset, print_log: bool = False):
        for metric in self.metrics:
            metric.retrieve_metrics(subset, print_log=print_log)

    def train_step(self, 
        batch, 
        screen_nan: bool = True, 
        loss_index: Optional[List[int]] = None # loss index for multi-loss models
    ):
        torch.set_grad_enabled(True)
        # create the batch dict on the device
        batch.to(self.device)
        batch_dict = batch.to_dict()
        # train and log the metrics
        self.train() # set training = true
        self.optimizer.zero_grad() # zero the gradient
        pred = self.model(batch_dict, training=True) # execute the model
        self.log_metrics('train', pred, batch_dict) 
        # set the loss and loss gradient
        loss = self.loss_fn(pred, batch_dict, {'epochs': self.global_step, 'training': True}, loss_index)
        loss.backward()
        # Print gradients for debugging purposes
        """
        for name, param in self.model.named_parameters():
            print(f"{name} requires grad: {param.requires_grad}")
            if param.requires_grad:
                print(f"Gradient of Loss w.r.t {name}: {param.grad}")
        """
        # clip gradient if requested
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        # check for nans
        no_nan = True
        if screen_nan:
            for param in self.model.parameters():
                if param.requires_grad and not torch.isfinite(param.grad).all():
                    no_nan = False
                    logging.info(f'!nan gradient!')
        # if no nans (no_nan==true), step forward
        if no_nan:
            # update the learning rate during warmup
            if self.global_step < self.warmup_steps:
                lr_scale = min(1.0, float(self.global_step + 1) / self.warmup_steps)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr_scale * self.lr
            # step the optimizer forward
            self.optimizer.step()
            # do ema if requested
            if self.ema and self.global_step >= self.ema_start:
                self.ema_model.update_parameters(self.model)
        # return the loss
        return loss.cpu().detach().numpy().item()

    def validate(self, val_loader):
        torch.set_grad_enabled(self.grad_enabled)
        self.eval()
        total_loss = 0.0
        for batch in val_loader:
            batch.to(self.device)
            batch_dict = batch.to_dict()
            if self.ema and self.global_step >= self.ema_start:
                pred = self.ema_model(batch_dict, training=False)
            else:
                pred = self.model(batch_dict, training=False)
            loss = (self.loss_fn(pred, batch_dict, {'epochs': self.global_step, 'training': False})).cpu().detach().numpy()
            total_loss += loss.item()
            self.log_metrics('val', pred, batch_dict)
        # return the loss
        return total_loss / len(val_loader)

    def fit(self, 
        train_loader, 
        val_loader, 
        epochs, 
        val_stride: int = 1, 
        screen_nan: bool = True,
        checkpoint_path: Optional[str] = 'checkpoint.pt',
        checkpoint_stride: int = 10,            
        bestmodel_path: Optional[str] = 'best_model.pth',
        print_stride: int = 1,
        subset_ratio: float = 1.0,
        subsample_loss_mode: Optional[int] = None,
    ):
        best_val_loss = float('inf')
        for epoch in range(1, epochs + 1):
            # train
            total_loss = 0
            if subset_ratio < 1.0:
                train_loader = self._get_subset_batches(train_loader, subset_ratio)
            for batch in train_loader:
                if subsample_loss_mode is not None:
                    loss_index = np.random.choice(len(self.losses), subsample_loss_mode)
                    loss = self.train_step(batch, screen_nan=screen_nan, loss_index=loss_index)
                else:
                    loss = self.train_step(batch, screen_nan=screen_nan, loss_index=None)
                total_loss += loss
                # step (sometimes)
                if self.scheduler:
                    if self.scheduler.__class__.__name__ == "OneCycleLR":
                        self.scheduler.step()
            avg_loss = total_loss / len(train_loader)
            # validate
            if print_stride > 0 and self.global_step % print_stride == 0:
                screen_output = True
            else:
                screen_output = False
            if epoch % val_stride == 0:
                val_loss = self.validate(val_loader)
                for pg in self.optimizer.param_groups:
                    lr_now = pg["lr"]
                    if screen_output:
                        print(f"##### Step: {self.global_step} Learning rate: {lr_now} #####")
                    logging.info(f"##### Step: {self.global_step} Learning rate: {lr_now} #####")
                if screen_output:
                    print(f'Epoch {epoch}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')
                logging.info(f'Epoch {epoch}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')
                self.retrieve_metrics('train', print_log=screen_output)
                self.retrieve_metrics('val', print_log=screen_output)
            # set learning rate
            if self.scheduler:
                if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.scheduler.step(val_loss)
                elif self.scheduler.__class__.__name__ != "OneCycleLR":
                    self.scheduler.step()
            # set checkpoints
            if epoch > val_stride and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(bestmodel_path, device=self.device)
            if checkpoint_path is not None and epoch % checkpoint_stride == 0:
                self.checkpoint(checkpoint_path)
            self.global_step += 1 

    # Function to subsample batches
    def _get_subset_batches(self, dataloader, subset_ratio: float):
        batches = list(dataloader)
        subset_size = int(subset_ratio * len(dataloader))
        if subset_size == 0: subset_size = 1
        indices = np.random.choice(len(batches), subset_size, replace=False)
        return [batches[i] for i in indices]

    def save_model(self, path: str, device: torch.device = torch.device('cpu')):
        if self.ema and self.global_step >= self.ema_start: 
            torch.save(self.ema_model.module.to(device), path)
            if device != self.device:
                self.ema_model.module.to(self.device)
        else:
            torch.save(self.model.to(device), path)
            if device != self.device:
                self.model.to(self.device)

    def checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_ema_state_dict': self.ema_model.state_dict() if self.ema and self.global_step >= self.ema_start else None,
            #'model_swa_state_dict': self.swa_model.state_dict() if self.swa and self.global_step >= self.swa_start else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, path)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model_state_dict'])
        if self.ema:
            self.ema_model.load_state_dict(state_dict['model_ema_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        if self.scheduler:
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])


