import numpy as np
import pandas as pd
import scanpy as sc
import sklearn
import torch
import cellrank as cr
from torchdyn.core import NeuralODE
import torch.nn.functional as F
import copy
from scCFM.models.cfm.cfm_module import torch_wrapper

def standardize_adata(adata, key):
    """
    standardize anndata and add a key to adata.layers
    """
    X_copy = adata.X.copy()
    adata.layers[key] = (X_copy - np.mean(X_copy, axis=0)) / np.std(X_copy, axis=0)
    
def add_keys_to_dict(metric_dict, metric_list):
    """
    Update a dictionary with a list
    """
    metric_list_zipped = dict(zip(metric_list[0], metric_list[1]))
    for m in metric_list_zipped:
        if m in metric_dict:
            metric_dict[m].append(metric_list_zipped[m])
        else:
            metric_dict[m] = []
            metric_dict[m].append(metric_list_zipped[m])
    return metric_dict

def real_reconstructed_cells_adata(model, 
                                   datamodule,
                                   process_amortized_adata=False,
                                   compute_umap=True, 
                                   log1p=False, 
                                   vae=True, 
                                   model_type="scvi"):
    """Create anndatas of latent cells and 

    Args:
        model (torch.nn.module): The autoencoder model
        datamodule (torch.utils.data.DataLoader): data loader with data to evaluate 

    Returns:
        dict: anndatas for z and concatenation of real and generated data
    """
    # Initialize the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Will contain library sizes and latent encodings 
    zs = []
    library_sizes = []
    
    if process_amortized_adata:
        real_cells = []
        recons_cells = []
        recons_cells_mu = []
        
    # Annotation for anndata 
    annot = {cond: [] for cond in datamodule.cond}

    with torch.no_grad():
        for _, batch in enumerate(datamodule.train_dataloader()):
            # Get the batch 
            batch["X"] = batch["X"].to(device)
            
            # If model external library size, collect it from the batch 
            library_size = batch["X"].sum(1).to(device)

            # Encode
            if vae:
                z, _, _ = model.encode(batch["X"]).values()
            else:
                z = model.encode(batch["X"])["z"]
            
            # Save the conditions 
            for cond in annot:
                annot[cond].append(batch[cond])
                
            # Collect latent observations
            zs.append(z)
            library_sizes.append(library_size)
            
            if process_amortized_adata:
                if model_type=="scvi":
                    c = library_size if model.model_library_size else torch.ones_like(library_size)
                    real_cells.append(batch["X"].cpu())
                    recons_cells.append(model.sample_decoder(z, library_size).cpu())
                    recons_cells_mu.append((model.decode(z)*c.unsqueeze(1)).cpu())
                
                else:
                    real_cells.append(torch.log1p(batch["X"].cpu()))
                    decoded_output = model.decode(z).cpu()
                    recons_cells.append(decoded_output)
                    recons_cells_mu.append(decoded_output)
    
    # Concatenate the results 
    zs = torch.cat(zs, dim=0).cpu().numpy()
    library_sizes = torch.cat(library_sizes).cpu()
    if process_amortized_adata:
        real_cells = torch.cat(real_cells, dim=0).cpu().numpy()
        recons_cells = torch.cat(recons_cells, dim=0).cpu().numpy()
        recons_cells_mu = torch.cat(recons_cells_mu, dim=0).cpu().numpy()

    # Concatenate annotations
    annot = {cond_key: torch.cat(cond_val).numpy() for cond_key, cond_val in annot.items()}
    annot_df = pd.DataFrame(annot)
    annot_df.columns = list(annot.keys())
    annot_df["log_library_size"] = torch.log(library_sizes).numpy()
    
    # Create anndata latent 
    adata_z = sc.AnnData(X=zs, 
                        obs=annot_df)
    sc.tl.pca(adata_z)
    if compute_umap:    
        sc.pp.neighbors(adata_z)
        sc.tl.umap(adata_z)
        
    # Create anndata mu 
    if process_amortized_adata:
        adata_mu = sc.AnnData(X=recons_cells_mu, 
                            obs=annot_df)
        
    # Create anndata real 
    if process_amortized_adata:
        amortized_and_real = np.concatenate([real_cells, recons_cells], axis=0)
        dataset_type = ["real" for _ in range(len(real_cells))] + ["generated_amortized" for _ in range(len(recons_cells))] 
        dataset_type_df = pd.DataFrame(dataset_type)
        dataset_type_df.columns = ["dataset_type"]
        annot_real_amortized = pd.concat([annot_df, annot_df], axis=0).reset_index(drop=True)

        adata_real_amortized = sc.AnnData(X=amortized_and_real, 
                                            obs=pd.concat([dataset_type_df, annot_real_amortized], axis=1))    

        if log1p:
            sc.pp.log1p(adata_real_amortized)

        sc.tl.pca(adata_real_amortized)
        if compute_umap:
            sc.pp.neighbors(adata_real_amortized)
            sc.tl.umap(adata_real_amortized)
        return dict(adata_real_amortized=adata_real_amortized, 
                    adata_z=adata_z, adata_mu=adata_mu)
    
    else:
        return adata_z
    
def add_velocity_to_adata(adata, model, device, model_library_size=True):
    """
    Compute and add velocity field to the data
    """
    # Put model in evaluation mode
    model.eval()
    velocities = []
    with torch.no_grad():
        for i, x in enumerate(adata.X):
            t = torch.tensor(adata.obs.experimental_time[i]).view(1, -1).float().to(device)
            x = torch.from_numpy(x).view(1, -1).float().to(device)
            if model_library_size:
                l = torch.from_numpy(np.array(adata.obs["log_library_size"][i])).view(1, -1).float().to(device)
                x = torch.cat([x,l], dim=1)
            dx_dt = model(x=x, t=t)
            velocities.append(dx_dt.cpu().numpy())
    velocities = np.concatenate(velocities, axis=0)

    if model_library_size:
        adata.layers["velocity"] = velocities[:, :-1]
    else:
        adata.layers["velocity"] = velocities[:, :-1]
    
def compute_velocity_projection(adata, xkey, vkey):
    """
    Project vector fiels onto low dimensional embedding 
    """
    vk = cr.kernels.VelocityKernel(adata,
                                  xkey=xkey, 
                                  vkey=vkey)
    return vk
    
def compute_trajectory(X_0,
                       l_0,
                       model,
                       idx2time, 
                       device, 
                       use_real_time, 
                       model_library_size=True):
    """
    Compute trajectory given the model 
    """
    node = NeuralODE(
        torch_wrapper(model.net), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    
    # Add library size to the state 
    if model_library_size:
        l_0 = l_0 if len(l_0.shape)==2 else l_0.unsqueeze(1)
        X_0 = torch.cat([X_0, l_0], dim=1)
        
    # Append first time point
    trajs = []
    times = []
    with torch.no_grad():
        # Save trajectory and real times
        trajs.append(X_0.detach().cpu())
        X_t = X_0
        times += [idx2time[0] for _ in range(X_t.shape[0])]
        for t in range(len(idx2time)-1):
            times += [idx2time[t+1] for _ in range(X_t.shape[0])]
            if use_real_time:
                time_range = torch.linspace(idx2time[t], idx2time[t+1], 1000)
                traj = node.trajectory(X_t.float().to(device),
                                            t_span=time_range,
                                            ).cpu()
            else:
                time_range = torch.linspace(t, t+1, 1000)
                traj = node.trajectory(X_t.float().to(device),
                                            t_span=time_range,
                                            ).cpu()
            # Keep only last value
            X_t = traj[-1]
            trajs.append(X_t.detach().cpu())
    return torch.cat(trajs, dim=0), times

def decode_trajectory(X_0,
                       l_0,
                       temporal_model,
                       vae,
                       idx2time, 
                       device, 
                       use_real_time, 
                       model_library_size=True, 
                       keep_time_d=False, 
                       append_last=True, 
                       model_type="vae"):
    """
    Compute trajectory given the model 
    """
    # Start node wrapper for simulation
    node = NeuralODE(
        torch_wrapper(temporal_model.net), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
    
    # Add library size to the state 
    if model_library_size:
        l_0 = l_0 if len(l_0.shape)==2 else l_0.unsqueeze(1)
        X_0 = torch.cat([X_0, l_0], dim=1)   
        
    # Arrays of results
    mu_trajs = []
    x_trajs = []
    
    # Times
    times = []
    
    with torch.no_grad():
        # Collect outputs at time 0
        mu_traj, x_traj = decode_state_lib_traj(vae, X_0, model_library_size, model_type=model_type)
        mu_trajs.append(mu_traj)
        x_trajs.append(x_traj)
        
        # Integrate through time
        X_t = X_0
        times += [idx2time[0] for _ in range(X_t.shape[0])]
        for t in range(len(idx2time)-1):
            time_range = torch.linspace(idx2time[t], idx2time[t+1], 1000)
            if append_last:
                times += [idx2time[t+1] for _ in range(X_t.shape[0])]
            else:
                times += list(torch.cat([idx2time[t]+time_range for _ in range(X_t.shape[0])]))
                
            if use_real_time:
                traj = node.trajectory(X_t.float().to(device),
                                            t_span=time_range)
            else:
                time_range = torch.linspace(t, t+1, 1000)
                traj = node.trajectory(X_t.float().to(device),
                                            t_span=time_range)
                
            if append_last: 
                X_t = traj[-1]
                mu_traj, x_traj = decode_state_lib_traj(vae, X_t, model_library_size, model_type=model_type)
                mu_trajs.append(mu_traj)
                x_trajs.append(x_traj)
            else:
                to_keep = torch.linspace(0,1000,5).to(torch.int)
                X_t = traj[:, to_keep, :]
                X_t = X_t.view(-1, X_t.shape[-1])
                mu_traj, x_traj = decode_state_lib_traj(vae, X_t, model_library_size, model_type=model_type)
                mu_trajs.append(mu_traj)
                x_trajs.append(x_traj)
    
    if not keep_time_d:
        return torch.cat(mu_trajs, dim=0), torch.cat(x_trajs, dim=0), times
    else:
        return torch.stack(mu_trajs, dim=1), torch.stack(x_trajs, dim=1), times
    
def decode_trajectory_single_step(X_0,
                                    l_0,
                                    t_0,
                                    temporal_model,
                                    vae, 
                                    model_library_size=True, 
                                    model_type="scvi"):
    """
    Compute trajectory given the model 
    """
    # Node for push forward
    node = NeuralODE(
        torch_wrapper(temporal_model.net),
        solver="dopri5", 
        sensitivity="adjoint", 
        atol=1e-4, 
        rtol=1e-4
        )
    
    # Add library size to the state 
    if model_library_size:
        l_0 = l_0 if len(l_0.shape)==2 else l_0.unsqueeze(1)
        X_0 = torch.cat([X_0, l_0], dim=1)   
    
    # Integrate through time
    with torch.no_grad():
        # Initial state
        X_t = X_0
        # Integrate a single step
        time_range = torch.linspace(t_0, t_0+1, 1000)
        traj = node.trajectory(X_t.float().to(vae.device),
                                t_span=time_range)
        mu_traj, x_traj = decode_state_lib_traj(vae, traj[-1], model_library_size, model_type)
    return mu_traj, x_traj, traj[-1]

def decode_state_lib_traj(model, X_t, model_library_size, model_type="vae"):
    """Perform decoding at a trajectory snapshot
    """
    if model_library_size and model_type!="geodesic_ae":
        mu_t_hat = F.softmax(model.decode(X_t[:, :-1]), dim=1)
        l_t_hat = torch.exp(X_t[:, -1])
        mu_t_hat = mu_t_hat * l_t_hat.unsqueeze(-1)
        X_t_hat = model.sample_decoder(X_t[:, :-1], l_t_hat).detach().cpu()
    elif model_type=="geodesic_ae":
        mu_t_hat = model.decode(X_t[:, :-1]).detach().cpu()
        X_t_hat = copy.deepcopy(mu_t_hat)
    else:
        mu_t_hat = torch.exp(model.decode(X_t))
        X_t_hat = model.sample_decoder(X_t).detach().cpu()
    return mu_t_hat.detach().cpu(), X_t_hat

def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    # Take only K nearest neighbors and the radius is the maximum of the knn distances 
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k):
    """
    Computes precision, recall, density, and coverage given two manifolds.
    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """
    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, 
                recall=recall,
                density=density, 
                coverage=coverage)

def standardize(tensor):
    """
    Standardize tensor across the rows
    """
    tensor = (tensor - tensor.mean(0)) / (tensor.std(0)+1e-6)
    return tensor
