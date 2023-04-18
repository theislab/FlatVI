from typing import Union, List
import numpy as np
import scanpy as sc

def adata_dataset(path, 
                  label_name: str = "experimental_time", 
                  multi_modal: bool = False, 
                  modality_selection_key: str = None,
                  use_pca: bool = True
                  ):
    """Read an adata in multiple layers
    """
    # Read AnnData 
    adata = sc.read_h5ad(path)
    # Subset highly variable genes 
    if "highly_variable" in adata.var.columns:
        adata = adata[:, adata.var.highly_variable]
    # Time labels
    labels = adata.obs[label_name].astype("category")
    # Individual labels 
    ulabels = labels.cat.categories
    # If multi-modal, define variable to select the modalities 
    if multi_modal:
        modality_selection =  adata.var[modality_selection_key]
    else: 
        modality_selection = None
    # Return either PCA reduction or the unreduced matrix 
    if use_pca:
        return adata.obsm["X_pca"].A, modality_selection, labels, ulabels 
    else:
        return adata.X.A, modality_selection, labels, ulabels


def load_dataset(path, 
                 label_name: str = "experimental_time", 
                 multi_modal: bool = False, 
                 modality_selection_key: str = None,
                 use_pca: bool = False):
    
    if path.endswith("h5ad"):
        return adata_dataset(path, 
                             label_name=label_name, 
                             multi_modal=multi_modal, 
                             modality_selection_key=modality_selection_key, 
                             use_pca=use_pca)
    else:
        raise NotImplementedError()
    