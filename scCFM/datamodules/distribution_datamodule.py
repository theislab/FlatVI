import os 

import numpy as np
from scCFM.datamodules.components.time_dataset import load_dataset
from functools import partial
from typing import Optional, List

import torch
from torch.utils.data import DataLoader, random_split, RandomSampler
from pytorch_lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader

class TrajectoryDataModule(LightningDataModule):
    pass_to_model = True
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
        num_workers: int = 0,
        seed=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        
        assert os.path.exists(path), "The data path does not exist"
        
        # Collect dataset 
        self.data, self.times = load_dataset(path=path, 
                                                x_layer=x_layer,
                                                time_key=time_key,
                                                use_pca=use_pca, 
                                                n_dimensions=n_dimensions)
        self.dim = self.data.shape[1]
        
        # Associate time to the process 
        self.idx2time = {idx: val for idx, val in enumerate(np.unique(self.times))}
        
        # Create a list of subset data
        self.timepoint_data = [
            self.data[self.times == lab].astype(np.float32) for lab in np.unique(self.times)
        ]
        self.split()

    def split(self):
        splitter = partial(
            random_split,
            lengths=self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(42),
        )
        self.split_timepoint_data = list(map(splitter, self.timepoint_data))

    def combined_loader(self, index, load_full=False):
        # Different timepoints may have different numbers of observations
        if load_full:
            n_samples = max([len(datasets) for datasets in self.timepoint_data])
            tp_dataloaders = [
                DataLoader(
                    dataset=datasets,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    drop_last=True,
                    sampler = RandomSampler(datasets, replacement=True, num_samples=n_samples)
                )
                for datasets in self.timepoint_data
            ]

        else:
            n_samples = max([len(datasets[index]) for datasets in self.split_timepoint_data])
            tp_dataloaders = [
                DataLoader(
                    dataset=datasets[index],
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    drop_last=True,
                    sampler = RandomSampler(datasets, replacement=True, num_samples=n_samples)
                )
                for datasets in self.split_timepoint_data
            ]
        return CombinedLoader(tp_dataloaders, mode="min_size")

    def train_dataloader(self, load_full=True):
        return self.combined_loader(0, load_full=load_full)

    def val_dataloader(self, load_full=False):
        return self.combined_loader(1, load_full=load_full)

    def test_dataloader(self, load_full=False):
        return self.combined_loader(2, load_full=load_full)
    