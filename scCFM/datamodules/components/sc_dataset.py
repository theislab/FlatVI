from typing import Union, Optional
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scanpy as sc

def adata_dataset(path: str, 
                  x_layer: str,
                  cond_keys: Union[str, list] = "experimental_time",
                  use_pca: bool = False,
                  n_dimensions: Optional[int] = None
                  ):
    """Loads AnnData from a path

    Args:
        path (str): the path to the AnnData object
        x_layer (str): the layer in AnnData to use 
        cond_key ([str, list], optional): the key(s) in adata.obs representing the conditions. Defaults to "experimental_time".
        use_pca (bool, optional): whether to use pca or gene counts. Defaults to False.
        n_dimensions (Optional[int], optional): number of dimensions to keep in PCA. Defaults to None.
    """
    
    # Read AnnData 
    adata = sc.read_h5ad(path)

    # Read labels, can be one or multiple
    if type(cond_keys) != list:
        cond_keys = [cond_keys]
    
    cond = {}
    for cond_key in cond_keys:
        cond[cond_key] = adata.obs[cond_key].to_numpy()
        if type(cond[cond_key][0]) == str:
            label_encoder = LabelEncoder()
            cond[cond_key] = label_encoder.fit_transform(cond[cond_key])
        
    # Return either PCA reduction or the unreduced matrix 
    if use_pca:
        if n_dimensions == None:
            n_dimensions = adata.obsm["X_pca"].shape[1]
        X = adata.obsm["X_pca"][:, n_dimensions]
    else:
        if x_layer in adata.layers:
            try:
                X = adata.layers[x_layer].A
            except:
                 X = adata.layers[x_layer]
        elif x_layer in adata.obsm:
            X = adata.obsm[x_layer]
        else:
            raise NotImplementedError
    return X, cond

def load_dataset(path: str, 
                 x_layer: str, 
                 cond_keys: str = "experimental_time", 
                 use_pca: bool = False, 
                 n_dimensions: int = None
                 ):
    """Wrapper around adata_dataset function to implement controls
    """
    return adata_dataset(path=path,
                            x_layer=x_layer,
                            cond_keys=cond_keys, 
                            use_pca=use_pca, 
                            n_dimensions=n_dimensions)

    