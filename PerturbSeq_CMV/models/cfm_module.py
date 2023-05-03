from typing import Any, List, Optional, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torchdyn.core import NeuralODE
from torch.optim import AdamW

from PerturbSeq_CMV.models.components.augmentation import (
    AugmentationModule,
    AugmentedVectorField,
    Sequential,
)
from PerturbSeq_CMV.models.components.distribution_distances import compute_distribution_distances
from PerturbSeq_CMV.models.components.optimal_transport import OTPlanSampler
from PerturbSeq_CMV.models.components.plotting import store_trajectories
from PerturbSeq_CMV.models.utils import get_wandb_logger


class CFMLitModule(LightningModule):
    """Conditional Flow Matching Module for training generative models and models over time."""

    def __init__(
        self,
        net: Any,
        datamodule: LightningDataModule,
        augmentations: AugmentationModule,
        ot_sampler: Optional[Union[str, Any]] = None,
        sigma_min: float = 0.1,
        avg_size: int = -1,
        leaveout_timepoint: int = -1,
        lr: float = 0.001, 
        weight_decay: float = 0.00001, 
        store_trajectories: bool = False
    ) -> None:
        """Initialize a conditional flow matching network either as a generative model or for a
        sequence of timepoints.
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=["net", "optimizer", "datamodule", "augmentations"], logger=False
        )
        # Contains the outputs of the validation step 
        self.state = []  
        # Aspect of the trajectory dataset 
        self.is_trajectory = datamodule.IS_TRAJECTORY
        self.dim = datamodule.dim
        # Net represents the neural network modelling the dynamics of the system (velocity)
        self.net = net(dim=datamodule.dim)
        # AugmentationModule instance computing forward augmentations of the input if required 
        self.augmentations = augmentations
        self.val_augmentations = AugmentationModule(
            l1_reg=1,
            l2_reg=1,
            squared_l2_reg=1,
        )
        # AugmentedVectorField object computes the dynamics over augmented vectors
        self.aug_net = AugmentedVectorField(self.net, self.augmentations.regs)
        self.val_aug_net = AugmentedVectorField(self.net, self.val_augmentations.regs)
        # Wrap neural ODE around the network
        self.node = NeuralODE(self.net)
        # Augmenter augments and AugmentationModule regularizes
        self.aug_node = Sequential(
            self.augmentations.augmenter,
            NeuralODE(self.aug_net, sensitivity="autograd"),
        )
        self.val_aug_node = Sequential(
            self.val_augmentations.augmenter,
            NeuralODE(self.val_aug_net, solver="rk4"),
        )
        # Define the optimizer and OT batch sampler 
        self.ot_sampler = ot_sampler
        if ot_sampler == "None":
            self.ot_sampler = None
        if isinstance(self.ot_sampler, str):
            # regularization taken for optimal Schrodinger bridge relationship
            self.ot_sampler = OTPlanSampler(method=ot_sampler, reg=2 * sigma_min**2)
        self.criterion = torch.nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.store_trajectories = False

    def forward_integrate(self, batch: Any, t_span: torch.Tensor):
        """Forward pass with integration over t_span intervals.
        (t, x, t_span) -> [x_t_span].
        """
        X = self.unpack_batch(batch)
        X_start = X[:, t_span[0], :]
        traj = self.node.trajectory(X_start, t_span=t_span)
        return traj

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        """Forward pass (t, x) -> dx/dt."""
        return self.net(t, x)

    def unpack_batch(self, batch):
        """Unpacks a batch of data to a single tensor."""
        if self.is_trajectory:
            return torch.stack(batch, dim=1)  # returns BxTxG
        return batch

    def preprocess_batch(self, X, training=False):
        """Given a full time-series trajectory, sample a time per batch observation and pair the 
        observation with the one coupled at the next time point. It returns random pair of (x0, x1)"""
        t_select = torch.zeros(1)
        if self.is_trajectory:
            batch_size, times, _ = X.shape
            if training and self.hparams.leaveout_timepoint > 0:
                # Select random experimental time except for the left-out timepoint
                t_select = torch.randint(times - 2, size=(batch_size,))
                t_select[t_select >= self.hparams.leaveout_timepoint] += 1
            else:
                t_select = torch.randint(times - 1, size=(batch_size,))
            x0 = []
            x1 = []
            for i in range(batch_size):
                ti = t_select[i]
                ti_next = ti + 1
                if training and ti_next == self.hparams.leaveout_timepoint:
                    ti_next += 1
                x0.append(X[i, ti])
                x1.append(X[i, ti_next])
            x0, x1 = torch.stack(x0), torch.stack(x1)
        else:
            batch_size, _ = X.shape
            # If no trajectory assume generate from standard normal
            x0 = torch.randn(batch_size, X.shape[1])
            x1 = X
        return x0, x1, t_select

    def step(self, batch: Any, training: bool = False):
        """Computes the loss on a batch of data."""
        X = self.unpack_batch(batch).to(self.device)
        x0, x1, t_select = self.preprocess_batch(X, training)
        t_select = t_select.to(self.device)

        if self.ot_sampler is not None:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)

        # Sample a t for the interpolation between observations
        if self.hparams.avg_size > 0:
            t = torch.rand(1, 1).repeat(X.shape[0], 1).to(self.device)
        else:
            t = torch.rand(X.shape[0], 1).to(self.device)
        
        # Sample interpolation between couples of observations 
        ut = x1 - x0
        mu_t = t * x1 + (1 - t) * x0
        sigma_t = self.hparams.sigma_min

        # If we are starting from right before the leaveout_timepoint then we
        # divide the target by 2
        if training and self.hparams.leaveout_timepoint > 0:
            ut[t_select + 1 == self.hparams.leaveout_timepoint] /= 2
            t[t_select + 1 == self.hparams.leaveout_timepoint] *= 2

        # t that network sees is incremented by first timepoint
        t = t + t_select[:, None]  # Use the batch time selected and sum to the interpolating time 
        x = mu_t + sigma_t * torch.randn_like(x0) # Sample from probability path 
        aug_x = self.aug_net(t, x, augmented_input=False)
        
        # Augmentations used as regularization, vector field for the criterion
        reg, vt = self.augmentations(aug_x) 
        return torch.mean(reg), self.criterion(vt, ut)

    def training_step(self, batch: Any, batch_idx: int):
        """Training step - computes vector field and computes regression loss
        """
        reg, mse = self.step(batch, training=True)
        loss = mse + reg
        prefix = "train"
        self.log_dict(
            {f"{prefix}/loss": loss, f"{prefix}/mse": mse, f"{prefix}/reg": reg},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        """Calls evaluation step 
        """
        return self.eval_step(batch, batch_idx, "val")
    
    def test_step(self, batch: Any, batch_idx: int):
        """Calls test step 
        """
        return self.eval_step(batch, batch_idx, "test")

    def eval_step(self, batch: Any, batch_idx: int, prefix: str):
        """Evaluation step
        """
        reg, mse = self.step(batch, training=False)
        loss = mse + reg
        self.log_dict(
            {f"{prefix}/loss": loss, f"{prefix}/mse": mse, f"{prefix}/reg": reg},
            on_step=False,
            on_epoch=True,
        )
        val_output = {"loss": loss, "mse": mse, "reg": reg, "x": self.unpack_batch(batch)}
        self.state.append(val_output)
        return val_output

    def on_validation_step_end(self):
        self.eval_epoch_end("val")    
    
    def on_test_step_end(self):
        self.eval_epoch_end("test")    
    
    def eval_epoch_end(self, prefix: str):
        # Collected outputs from the previous batch
        outputs = self.state
        if self.is_trajectory and prefix == "test" and isinstance(outputs[0]["x"], list):
            # x is jagged if doing a trajectory
            x = outputs[0]["x"]
            ts = len(x)
            x0 = x[0]
            x_rest = x[1:]

        elif self.is_trajectory:
            v = {k: torch.cat([d[k] for d in outputs]) for k in ["x"]}
            # Simply collect information abuout the trajectory and the observations along it 
            x = v["x"]
            ts = x.shape[1]
            x0 = x[:, 0, :]
            x_rest = x[:, 1:]

        else:
            v = {k: torch.cat([d[k] for d in outputs]) for k in ["x"]}
            x = v["x"]
            # Sample some random points for the plotting function
            rand = torch.randn_like(x)
            x = torch.stack([rand, x], dim=1)
            ts = x.shape[1]
            x0 = x[:, 0, :]
            x_rest = x[:, 1:]

        if self.current_epoch == 0:
            # skip epoch zero for numerical integration reasons
            return

        # Build a trajectory
        t_span = torch.linspace(0, 1, 101)  # predictions from 0 to 1
        aug_dims = self.val_augmentations.aug_dims
        regs = []
        trajs = []
        for i in range(ts - 1):  # ts represent the external time and is used to select obs
            if self.is_trajectory and prefix == "test":
                x_start = x[i]
            else:
                x_start = x[:, i, :]
            _, aug_traj = self.val_aug_node(x_start, t_span + i) # Push forward observations 
            aug, traj = aug_traj[-1, :, :aug_dims], aug_traj[-1, :, aug_dims:] # Only pick last observations in the traj
            trajs.append(traj)
            # Mean regs over batch dimension
            regs.append(torch.mean(aug, dim=0).detach().cpu().numpy())
        regs = np.stack(regs).mean(axis=0)
        names = [f"{prefix}/{name}" for name in self.val_augmentations.names]
        self.log_dict(dict(zip(names, regs)))

        # Evaluate the fit - Compare the predicted trajectories (from teacher forcing) with the real ones
        if self.is_trajectory and prefix == "test" and isinstance(outputs[0]["x"], list):
            names, dists = compute_distribution_distances(trajs[:-1], x_rest[:-1])
        else:
            names, dists = compute_distribution_distances(trajs, x_rest)
        names = [f"{prefix}/{name}" for name in names]
        d = dict(zip(names, dists))
        if self.hparams.leaveout_timepoint >= 0:
            to_add = {
                f"{prefix}/t_out/{key.split('/')[-1]}": val
                for key, val in d.items()
                if key.startswith(f"{prefix}/t{self.hparams.leaveout_timepoint}")
            }
            d.update(to_add)
        self.log_dict(d)

        if prefix == "test" and self.store_trajectories:
            store_trajectories(x, self.net)
        
        # Reset the state before the next epoch
        self.state = []

    def configure_optimizers(self):
        return AdamW(params=self.parameters(), 
                     lr=self.lr, 
                     weight_decay=self.weight_decay)
        