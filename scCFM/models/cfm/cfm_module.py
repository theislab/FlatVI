from typing import Any, Optional, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torchdyn.core import NeuralODE
from torch.optim import Adam

from scCFM.models.cfm.components.distribution_distances import compute_distribution_distances
from scCFM.models.cfm.components.optimal_transport import OTPlanSampler
from scCFM.models.cfm.utils import pad_t_like_x

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


class CFMLitModule(LightningModule):
    """Conditional Flow Matching Module for training generative models and models over time."""

    def __init__(
        self,
        in_dim: Any,
        net: Any,
        datamodule: LightningDataModule,
        ot_sampler: Optional[Union[str, Any]] = None,
        sigma: float = 0.1,
        lr: float = 0.001, 
        weight_decay: float = 0.00001, 
        use_real_time: bool = True,
        antithetic_time_sampling: bool = True
    ) -> None:
        """Args:
            autoencoder (torch.nn.Module): a pre-trained autoencoder model 
            net (torch.nn.Module): Network parametrizing the velocity 
            datamodule (LightningDataModule): Wrapper around data loaders  
            ot_sampler (Optional[Union[str, Any]], optional): Type of optimal transport . Defaults to None.
            sigma (float, optional): Optimal transport parameter. Defaults to 0.1.
            lr (float, optional): learning rate for optimization. Defaults to 0.001.
            weight_decay (float, optional): weight decay for optimization. Defaults to 0.00001.
            use_real_time (bool): use real time instead of fictitious time 
            freeze_autoencoder (bool): if autoencoder is present, whether its weights should be frozen 
        """
        super().__init__()
        
        # State for evaluation
        self.state = []
    
        # Attributes of the trajectory dataset 
        self.datamodule = datamodule
        self.dim = datamodule.dim
        self.idx2time = datamodule.idx2time
        self.lr = lr
        self.weight_decay = weight_decay
        self.sigma = sigma
        self.use_real_time = use_real_time
        self.antithetic_time_sampling = antithetic_time_sampling
        
        # Net represents the neural network modelling the dynamics of the system (velocity)
        in_dim = datamodule.dim if self.dim==None else self.dim
        self.net = net(dim=in_dim)
        
        # Wrap neural ODE around the network
        self.node = NeuralODE(torch_wrapper(self.net),
                              solver="dopri5", 
                              sensitivity="adjoint", 
                              atol=1e-4, 
                              rtol=1e-4)
        
        # Define the optimizer and OT batch sampler 
        self.ot_sampler = ot_sampler
        if ot_sampler == "None":
            self.ot_sampler = None
        if isinstance(self.ot_sampler, str):
            self.ot_sampler = OTPlanSampler(method=ot_sampler)
        self.criterion = torch.nn.MSELoss()
        
    def unpack_batch(self, batch):
        """Unpacks a batch of data to a single tensor."""
        return torch.stack(batch, dim=1)  # returns BxTxG
    
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
        t = pad_t_like_x(t, x)
        return self.net(torch.cat([x, t], dim=1))

    def sample_times(self, batch_size):
        if self.antithetic_time_sampling:
            t0 = np.random.uniform(0, 1 / batch_size)
            times = torch.arange(t0, 1.0, 1.0 / batch_size)
        else:
            times = torch.rand(batch_size)
        return times

    def compute_sigma_t(self, t):
        return self.sigma
    
    def compute_mu_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def sample_xt(self, x0, x1, t, epsilon):
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1):
        return x1 - x0

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1):
        t = self.sample_times(x0.shape[0]).type_as(x0)
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1)
        return t, xt, ut
    
    def preprocess_batch(self, X):
        """Given a full time-series trajectory, sample a time per batch observation and pair the 
        observation with the one coupled at the next time point. It returns random pair of (x0, x1)"""
        # Batch size and time shape
        _, n_times, _ = X.shape
        ts = []  # Contains the sampled uniform times
        xts = []  # Contains the sampled Xs
        uts = []  # Contains the objective velocities 
        
        for t_select in range(n_times-1):
            x0 = X[:, t_select, :]
            x1 = X[:, t_select + 1, :]
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)
            t, xt, ut = self.sample_location_and_conditional_flow(x0, x1)
            
            if self.use_real_time:
                t1_minus_t0 = self.idx2time[t_select + 1] - self.idx2time[t_select] 
                t = pad_t_like_x(t * t1_minus_t0 + self.idx2time[t_select], x0)
            else:
                t = pad_t_like_x(t + t_select, x0)
            
            ts.append(t)
            xts.append(xt)
            uts.append(ut)
        
        # Concatenate results 
        t = torch.cat(ts)
        xt = torch.cat(xts)
        ut = torch.cat(uts)
        return t, xt, ut
    
    def step(self, batch: Any, training: bool = False):
        """Computes the loss on a batch of data."""
        X = self.unpack_batch(batch).to(self.device)
        t, xt, ut = self.preprocess_batch(X)
        vt = self.net(torch.cat([xt, t], dim=1) )
        
        return self.criterion(vt, ut)

    def training_step(self, batch: Any, batch_idx: int):
        """Training step - computes vector field and computes regression loss
        """
        loss = self.step(batch, training=True)
        self.log_dict(
            {f"train/loss": loss },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        """Evaluation step
        """
        # Reset the state before the next epoch
        loss = self.step(batch, training=False)
        self.log_dict(
            {f"val/loss": loss},
            on_step=False,
            on_epoch=True,
        )
        self.state.append(self.unpack_batch(batch).cpu())

    def on_validation_epoch_end(self):
        self.eval_epoch_end("val")       
    
    def eval_epoch_end(self, prefix: str):
        """Final evaluation post epoch
        """
        x = torch.cat(self.state, dim=0) 
        # Simply collect information abuout the trajectory and the observations along it 
        ts = x.shape[1]  # times
        x_rest = x[:, 1:]
            
        # skip epoch zero for numerical integration reasons
        if self.current_epoch == 0:
            return

        # Build a trajectory
        t_span = torch.linspace(0, 1, 101)  # predictions from 0 to 1            

        # Augmentations for regularization
        trajs = []
        for i in range(ts - 1):  # ts represent the external time and is used to select obs
            x_start = x[:, i, :].to(self.device)
            
            # Piecewise integration
            if self.use_real_time:  
                t_select = self.idx2time[i]
                t1_minus_t0 = self.idx2time[i+1] - self.idx2time[i]
                traj = self.node.trajectory(x_start, t_span * t1_minus_t0  + t_select) 
            else:
                t_select = i
                traj = self.node.trajectory(x_start, t_span + t_select) 

            # Append trajectory
            traj = traj[-1]    
            trajs.append(traj.detach().cpu())
        
        # Evaluate the fit - Compare the predicted trajectories (from teacher forcing) with the real ones
        names, dists = compute_distribution_distances(trajs, x_rest.cpu())
        
        # Log results
        names = [f"{prefix}/{name}" for name in names]
        d = dict(zip(names, dists))
        self.log_dict(d)
        
        # Reset state
        self.state = []
            
    def configure_optimizers(self):
        return Adam(params=self.parameters(), 
                     lr=self.lr)
        