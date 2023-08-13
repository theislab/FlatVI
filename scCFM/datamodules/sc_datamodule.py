import os
import numpy as np
from scCFM.datamodules.components.sc_dataset import load_dataset
from functools import partial
from typing import Optional, List
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from pytorch_lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader

class CellDataset(Dataset):
    def __init__(self, data, batch):
        self.data = data
        self.batch = batch

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.tensor(self.data[idx])
        batch = torch.tensor(self.batch[idx])
        return {"X": X.to(torch.float32), "cond": batch.to(torch.float32)}

class scDataModule(LightningDataModule):
    pass_to_model = True

    def __init__(
        self,
        path: str,
        x_layer: str,
        cond_key: str, 
        use_pca: bool, 
        n_dimensions: Optional[int] = None, 
        train_val_test_split: List = [0.8, 0.2],
        batch_size: Optional[int] = 64,
        num_workers: int = 0
    ):
        super().__init__()
                
        assert os.path.exists(path), "The data path does not exist"
        
        # Collect dataset 
        self.data, self.cond = load_dataset(path=path, 
                                            x_layer=x_layer,
                                            cond_key=cond_key,
                                            use_pca=use_pca, 
                                            n_dimensions=n_dimensions)
        self.in_dim = self.data.shape[1]
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Associate time to the process 
        self.idx2cond = {idx: val for idx, val in enumerate(np.unique(self.cond))}
        
        # Create dataset 
        self.dataset = CellDataset(self.data, self.cond)
        
        # Create a list of subset data
        self.split()

    def split(self):
        self.split_data = random_split(self.dataset, self.train_val_test_split)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.split_data[0],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.split_data[1],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.split_data[2],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False
        )
        