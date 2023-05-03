import os 

import numpy as np
from PerturbSeq_CMV.datamodules.components.time_dataset import load_dataset
from functools import partial
from typing import Optional, List, Union

import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader

class TrajectoryDataModule(LightningDataModule):
    pass_to_model = True
    IS_TRAJECTORY = True
    
    def __init__(
        self,
        path: str = "",
        train_val_test_split: List = [0.8],
        max_dim: Optional[int] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed=None,
        label_name: str = "experimental_time", 
        multi_modal: bool = False, 
        modality_selection_key: str = None,
        use_pca: bool = False
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        
        assert os.path.exists(path), "The data path does not exist"
        
        # Collect dataset 
        self.data, self.modality_selection, self.labels, self.ulabels = load_dataset(path=path, 
                                                                                        label_name=label_name, 
                                                                                        multi_modal=multi_modal, 
                                                                                        modality_selection_key=modality_selection_key,
                                                                                        use_pca=use_pca)

        # Subset to maximum dimension if necessary 
        if max_dim:
            self.data = self.data[:, :max_dim]

        self.dim = self.data.shape[-1]

        # Create a list of subsetted data 
        self.timepoint_data = [
            self.data[self.labels == lab].astype(np.float32) for lab in self.ulabels
        ]
        self.split()

    def split(self):
        """split requires self.hparams.train_val_test_split, timepoint_data, ulabels."""
        splitter = partial(
            random_split,
            lengths=self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(42),
        )
        self.split_timepoint_data = list(map(splitter, self.timepoint_data))

    def combined_loader(self, index, shuffle=False, load_full=False):
        if load_full:
            tp_dataloaders = [
                DataLoader(
                    dataset=datasets,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    shuffle=False,
                    drop_last=True,
                )
                for datasets in self.timepoint_data
            ]
        else:
            tp_dataloaders = [
                DataLoader(
                    dataset=datasets[index],
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    shuffle=shuffle,
                    drop_last=True,
                )
                for datasets in self.split_timepoint_data
            ]
        return CombinedLoader(tp_dataloaders, mode="min_size")

    def train_dataloader(self):
        return self.combined_loader(0, shuffle=True, load_full=True)

    def val_dataloader(self):
        return self.combined_loader(1, shuffle=False, load_full=True)

    def test_dataloader(self):
        return self.combined_loader(2, shuffle=False, load_full=True)
    