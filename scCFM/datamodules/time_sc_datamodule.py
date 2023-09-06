import os 

import numpy as np
from scCFM.datamodules.components.sc_dataset import load_dataset
from typing import Optional, List

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from pytorch_lightning import LightningDataModule

class TrajectoryDataset(Dataset):
    def __init__(
        self,
        path: str,
        x_layer: str,
        time_key: str, 
        use_pca: bool, 
        n_dimensions: Optional[int] = None):
        
        super().__init__()
        assert os.path.exists(path), "The data path does not exist"

        # Collect dataset 
        self.data, self.times = load_dataset(path=path, 
                                                x_layer=x_layer,
                                                cond_keys=time_key,
                                                use_pca=use_pca, 
                                                n_dimensions=n_dimensions)
        self.times = self.times["experimental_time"]
        
        # Dimensionality of cells
        self.dim = self.data.shape[1]
        
        # Initialize times from lower to higher
        self.times_unique = sorted(np.unique(self.times))
        
        # Associate index to unique time values 
        self.idx2time = {idx: val for idx, val in enumerate(self.times_unique)}
        
        # Create a list of subset data
        self.timepoint_data = [
            self.data[self.times == lab].astype(np.float32) for lab in self.times_unique
        ]
        
    def __len__(self):
        return max([i.shape[0] for i in self.timepoint_data])
    
    def __getitem__(self, idx):
        batch = []
        for tp in range(len(self.times_unique)):
            data = self.timepoint_data[tp]
            idx = np.random.choice(len(data))
            batch.append(torch.tensor(data[idx]))
        return batch
         
         
class TrajectoryDataModule(LightningDataModule):
    IS_TRAJECTORY = True
    
    def __init__(
        self,
        path: str,
        x_layer: str,
        time_key: str, 
        use_pca: bool, 
        n_dimensions: Optional[int] = None, 
        train_val_test_split: List = [0.8, 0.2],
        batch_size: Optional[int] = 64,
        num_workers: int = 0):
        
        super().__init__()
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = TrajectoryDataset(path,
                                        x_layer,
                                        time_key, 
                                        use_pca, 
                                        n_dimensions)
        
        self.dim = self.dataset.dim
        self.idx2time = self.dataset.idx2time
        self.splits = random_split(self.dataset,
                                   lengths=self.train_val_test_split,
                                   generator=torch.Generator().manual_seed(42))
    
    def train_dataloader(self):
        return DataLoader(
                    dataset=self.splits[0],
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    drop_last=True
                )

    def val_dataloader(self):
        return DataLoader(
                    dataset=self.splits[1],
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    drop_last=True
                )

    def test_dataloader(self):
        return DataLoader(
                    dataset=self.splits[2],
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    drop_last=True
                )
