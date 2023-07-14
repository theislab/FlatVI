from typing import Union, List, Optional
import numpy as np
import scanpy as sc

def adata_dataset(path: str, 
                  x_layer: str,
                  time_key: str = "experimental_time", 
                  use_pca: bool = False,
                  n_dimensions: Optional[int] = None
                  ):
    """Loads AnnData from a path

    Args:
        path (str): the path to the AnnData object
        x_layer (str): the layer in AnnData to use 
        time_key (str, optional): the key in adata.obs representing the time. Defaults to "experimental_time".
        use_pca (bool, optional): whether to use pca or gene counts. Defaults to False.
        n_dimensions (Optional[int], optional): number of dimensions to keep in PCA. Defaults to None.
    """
    
    # Read AnnData 
    adata = sc.read_h5ad(path)

    # Time labels
    times = np.array(adata.obs[time_key])
        
    # Return either PCA reduction or the unreduced matrix 
    if use_pca:
        return adata.obsm["X_pca"][:, n_dimensions], times 
    else:
        return adata.layers[x_layer].A, times

def load_dataset(path: str, 
                 x_layer: str, 
                 time_key: str = "experimental_time", 
                 use_pca: bool = False, 
                 n_dimensions: int = None
                 ):
    """Wrapper around adata_dataset function to implement controls
    """
    
    if path.endswith("h5ad"):
        return adata_dataset(path=path,
                             x_layer=x_layer,
                             time_key=time_key, 
                             use_pca=use_pca, 
                             n_dimensions=n_dimensions)
    else:
        raise NotImplementedError()
    