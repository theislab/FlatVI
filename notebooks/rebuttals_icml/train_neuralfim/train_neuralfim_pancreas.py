import scanpy as sc
import numpy as np
import phate 
import os
import scipy as sp
from scipy.spatial.distance import pdist, squareform
from data import train_valid_loader_from_pc
from model import AEDist
import torch
from transformations import LogTransform, NonTransform, StandardScaler, \
    MinMaxScaler, PowerTransformer, KernelTransform

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import Dataset

import sys 
sys.path.insert(0, "/home/icb/alessandro.palma/environment/flatvi_baselines/phate_fim/src")
from src.models.lit_encoder import LitAutoencoder

class torch_dataset(Dataset):
    def __init__(self, X, Y) -> None:
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        target = self.Y[index, :]
        sample = self.X[index, :]
        return sample, target
    
# Read pancreas AnnData 
adata = sc.read_h5ad("/home/icb/alessandro.palma/environment/scCFM/project_dir/data/pancreas/processed/pancreas.h5ad")

# Collect expression matrix 
X_expression = np.array(adata.X.copy().todense())  # log-gexp 

# Initialize PHATE 
phate_op = phate.PHATE() 
phate_coords = phate_op.fit_transform(X_expression)
phate_coords = sp.stats.zscore(phate_coords) 

# Initialize tensors and data loader 
X_expression = torch.from_numpy(X_expression)
phate_coords = torch.from_numpy(phate_coords)
train_dataset = torch_dataset(X_expression, phate_coords)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=256, shuffle=True)

# Callable dict 
class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Hparams 
args = Args(
    run_name=None,
    dataset="tree",
    n_obs=1600,
    n_dim=2000,
    batch_size=256,
    knn=5,
    max_epochs=100,
    wandb=False,
    encoder_layer=[256, 100, 10],
    decoder_layer=[10, 100, 256],
    activation="ReLU",
    lr=0.0001,
    kernel_type="phate",
    loss_emb=False,
    loss_dist=True,
    loss_rec=True,
    scale=0.0005,
    inference=False,
    inference_obs=1600
)

# Get the dictionary of the variables 
dict_args = vars(args)

logger = WandbLogger(project="fim_phate", name=args.run_name)

# Initialize trainer 
trainer = Trainer.from_argparse_args(
    args, accelerator="gpu", devices=1, logger=logger
)

# Embedding dimension 
emb_dim = args.encoder_layer[-1]

# Autoencoder 
model = LitAutoencoder(input_dim=args.n_dim, emb_dim=emb_dim, **dict_args)
trainer.fit(model, train_dataloaders=train_loader)

# Save the model 
torch.save(model.state_dict(), "/home/icb/alessandro.palma/environment/scCFM/project_dir/baselines/neuralfim/model_ckpt.pt")
