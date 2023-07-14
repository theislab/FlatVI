from typing import Any, List, Optional, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torchdyn.core import NeuralODE
from torch.optim import AdamW

from scCFM.models.cfm.components.distribution_distances import compute_distribution_distances
from scCFM.models.cfm.components.optimal_transport import OTPlanSampler
from scCFM.models.cfm.components.plotting import store_trajectories


class CFMLitModule(LightningModule):
    """Conditional Flow Matching Module for training generative models and models over time."""

    def __init__(
        self,
        net: Any,
        datamodule: LightningDataModule,
        ot_sampler: Optional[Union[str, Any]] = None,
        sigma_min: float = 0.1,
        leaveout_timepoint: int = -1,
        lr: float = 0.001, 
        weight_decay: float = 0.00001, 
        store_trajectories: bool = False,
        use_real_time: bool = True,
    ) -> None:
        """
        Args:
            net (torch.nn.Module): Network parametrizing the velocity 
            datamodule (LightningDataModule): Wrapper around data loaders  
            ot_sampler (Optional[Union[str, Any]], optional): Type of optimal transport . Defaults to None.
            sigma_min (float, optional): Optimal transport parameter. Defaults to 0.1.
            leaveout_timepoint (int, optional): What timepoint to leave out. Defaults to -1.
            lr (float, optional): learning rate for optimization. Defaults to 0.001.
            weight_decay (float, optional): weight decay for optimization. Defaults to 0.00001.
            store_trajectory (bool): store trajectory of points 
            use_real_time (bool):  
        """
        super().__init__()
        # Save hyperparameters of the model class
        self.save_hyperparameters(
            ignore=["net", "optimizer", "datamodule", "augmentations"], logger=False
        )
        
        # Contains the outputs of the validation step 
        self.state = []  
    
        # Attributes of the trajectory dataset 
        self.is_trajectory = datamodule.IS_TRAJECTORY
        self.dim = datamodule.dim
        self.idx2time = datamodule.idx2time
        
        # Net represents the neural network modelling the dynamics of the system (velocity)
        self.net = net(dim=datamodule.dim)
        
        # Wrap neural ODE around the network
        self.node = NeuralODE(self.net)
        
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
        self.use_real_time =  use_real_time 

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
            
            # Select random experimental time except for the left-out timepoint
            if training and self.hparams.leaveout_timepoint > 0:
                t_select = torch.randint(times - 2, size=(batch_size,))
                t_select[t_select >= self.hparams.leaveout_timepoint] += 1
            else:
                t_select = torch.randint(times - 1, size=(batch_size,))
                
            x0 = []
            x1 = []
            t1_minus_t0 = []
            for i in range(batch_size):
                ti = t_select[i]
                ti_next = ti + 1
                if training and ti_next == self.hparams.leaveout_timepoint:
                    ti_next += 1
                x0.append(X[i, ti])
                x1.append(X[i, ti_next])
                # Initialize the difference between adjacent time points for interpolation
                t1_minus_t0.append(self.idx2time[ti_next.item()]-
                                   self.idx2time[ti.item()])
            x0, x1, t1_minus_t0 = torch.stack(x0), torch.stack(x1), torch.tensor(t1_minus_t0).to(self.device)
        else:
            raise NotImplementedError
        return x0, x1, t_select, t1_minus_t0

    def step(self, batch: Any, training: bool = False):
        """Computes the loss on a batch of data."""
        X = self.unpack_batch(batch).to(self.device)
        x0, x1, t_select, t1_minus_t0 = self.preprocess_batch(X, training)
        t_select = t_select.to(self.device)
        
        if self.ot_sampler is not None:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)

        # Sample a t for the interpolation between observations
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
        
        # Collect real time in case
        if self.use_real_time and t1_minus_t0!=None:
            t_select = torch.tensor([self.idx2time[i.item()] for i in t_select]).to(self.device)
        else:
            t1_minus_t0 = torch.ones(1)
        
        # t that network sees is incremented by first timepoint
        t = t * t1_minus_t0[:, None]  + t_select[:, None]  # Use the batch time selected and sum to the interpolating time 
        t = t.to(torch.float32)
        x = mu_t + sigma_t * torch.randn_like(x0) # Sample from probability path 
        vt = self.net(t, x)
        
        return self.criterion(vt, ut)

    def training_step(self, batch: Any, batch_idx: int):
        """Training step - computes vector field and computes regression loss
        """
        loss = self.step(batch, training=True)
        prefix = "train"
        self.log_dict(
            {f"{prefix}/loss": loss },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        """Calls evaluation step 
        """
        return self.eval_step(batch, batch_idx, "val")

    def eval_step(self, batch: Any, batch_idx: int, prefix: str):
        """Evaluation step
        """
        # Reset the state before the next epoch
        loss = self.step(batch, training=False)
        self.log_dict(
            {f"{prefix}/loss": loss},
            on_step=False,
            on_epoch=True,
        )
        val_output = {"loss": loss, "x": self.unpack_batch(batch).cpu()}
        self.state.append(val_output)
        return val_output

    def on_validation_epoch_end(self):
        self.eval_epoch_end("val")       
    
    def eval_epoch_end(self, prefix: str):
        """Final evaluation post epoch
        """
        if self.is_trajectory:
            v = {k: torch.cat([d[k] for d in self.state]) for k in ["x"]}
            # Simply collect information abuout the trajectory and the observations along it 
            x = v["x"]
            del v
            ts = x.shape[1]  # times
            x_rest = x[:, 1:]

        else:
            raise NotImplementedError
            
        # skip epoch zero for numerical integration reasons
        if self.current_epoch == 0:
            return

        # Build a trajectory
        t_span = torch.linspace(0, 1, 101)  # predictions from 0 to 1            

        # Augmentations for regularization
        trajs = []
        for i in range(ts - 1):  # ts represent the external time and is used to select obs
            # If use the real time you have to prompt it to the algorithm
            if self.use_real_time:
                t_select = self.idx2time[i]
                t1_minus_t0 = self.idx2time[i+1] - self.idx2time[i]
            else:
                t1_minus_t0 = torch.ones(1)
            
            # Piecewise integration
            x_start = x[:, i, :].cuda()
            _, traj = self.node(x_start, t_span * t1_minus_t0  + t_select) # Push forward observations
            traj = traj[-1]
            
            trajs.append(traj.detach().cpu())
        
        # Evaluate the fit - Compare the predicted trajectories (from teacher forcing) with the real ones
        names, dists = compute_distribution_distances(trajs, x_rest.cpu())
        
        # Log results
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
        
        # Reset state
        self.state = []

    def configure_optimizers(self):
        return AdamW(params=self.parameters(), 
                     lr=self.lr, 
                     weight_decay=self.weight_decay)
        