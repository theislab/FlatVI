import scanpy as sc
import numpy as np
import phate 
from scipy.spatial.distance import pdist, squareform
from data import train_valid_loader_from_pc
from model import AEDist
import torch
from transformations import LogTransform, NonTransform, StandardScaler, \
    MinMaxScaler, PowerTransformer, KernelTransform

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# Read data 
adata = sc.read_h5ad("/home/icb/alessandro.palma/environment/scCFM/project_dir/data/pancreas/processed/pancreas.h5ad")

# Collect gene expression 
X_expression = np.array(adata.X.copy().todense())

# Initialize and compute PHATE embedding 
phate_op = phate.PHATE()
phate_coords = phate_op.fit_transform(X_expression)

# Mutual distance as a squared matrix 
phate_D = squareform(pdist(phate_coords))
dist_std = np.std(phate_D.flatten())

trainloader, valloader, mean, std = train_valid_loader_from_pc(
            X_expression, # <---- Pointcloud
            phate_D, # <---- Distance matrix to match
            batch_size=256,
            train_valid_split=0.8,
            shuffle=True,
            seed=42, return_mean_std=True, componentwise_std=False)

model = AEDist(
            dim=X_expression.shape[1],
            emb_dim=10,
            layer_widths=[256, 128, 64],
            activation_fn=torch.nn.ReLU(),
            dist_reconstr_weights=[0.9, 0.1, 0.],
            pp=NonTransform(),
            lr=0.001,
            weight_decay=0.0001,
            batch_norm=True,
            dist_recon_topk_coords=0,
            use_dist_mse_decay=False,
            dist_mse_decay=0.,
            dropout=0.,
            cycle_weight=0.,
            cycle_dist_weight=0.,
            mean=mean,
            std=std,
            dist_std=dist_std)

path_dir = "/home/icb/alessandro.palma/environment/scCFM/project_dir/baselines/gaga/run"
path_model = "/home/icb/alessandro.palma/environment/scCFM/project_dir/baselines/gaga/model"

logger =  WandbLogger(project="gaga_phate", name=None)

checkpoint_callback = ModelCheckpoint(
    dirpath=path_dir,  # Save checkpoints in wandb directory
    filename=path_model,
    save_top_k=1,
    monitor='train_loss_step',  # Model selection based on validation loss
    mode='min',  # Minimize validation loss,
    every_n_train_steps=10000
)

trainer = Trainer(
    logger=logger,
    max_epochs=10,
    accelerator='cuda',
    callbacks=[checkpoint_callback],
    log_every_n_steps=100,
)

trainer.fit(
    model=model,
    train_dataloaders=trainloader)
