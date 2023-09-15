import numpy as np
import pandas as pd
import scanpy as sc
import torch
import cellrank as cr
from torchdyn.core import NeuralODE
import torch.nn.functional as F
from scCFM.models.cfm.cfm_module import torch_wrapper

def real_reconstructed_cells_adata(model, 
                                   datamodule,
                                   process_amortized_adata=False,
                                   compute_umap=True, 
                                   log1p=False, 
                                   vae=True):
    """Create anndatas of latent cells and 

    Args:
        model (torch.nn.module): The autoencoder model
        datamodule (torch.utils.data.DataLoader): data loader with data to evaluate 

    Returns:
        dict: anndatas for z and concatenation of real and generated data
    """
    # Lists containing results of encoding/decoding 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
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
                c = library_size if model.model_library_size else torch.ones_like(library_size)
                real_cells.append(batch["X"].cpu())
                recons_cells.append(model.sample_decoder(z, library_size).cpu())
                recons_cells_mu.append((model.decode(z)*c.unsqueeze(1)).cpu())
    
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
    
def compute_velocity_projection(adata, xkey, vkey, basis):
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
                       model_library_size=True):
    """
    Compute trajectory given the model 
    """
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
        mu_traj, x_traj = decode_state_lib_traj(vae, X_0, model_library_size)
        mu_trajs.append(mu_traj)
        x_trajs.append(x_traj)
        
        # Integrate through time
        X_t = X_0
        times += [idx2time[0] for _ in range(X_t.shape[0])]
        for t in range(len(idx2time)-1):
            times += [idx2time[t+1] for _ in range(X_t.shape[0])]
            if use_real_time:
                time_range = torch.linspace(idx2time[t], idx2time[t+1], 1000)
                traj = node.trajectory(X_t.float().to(device),
                                            t_span=time_range,
                                            )
            else:
                time_range = torch.linspace(t, t+1, 1000)
                traj = node.trajectory(X_t.float().to(device),
                                            t_span=time_range,
                                            )
            X_t = traj[-1]
            mu_traj, x_traj = decode_state_lib_traj(vae, X_t, model_library_size)
            mu_trajs.append(mu_traj)
            x_trajs.append(x_traj)
    return torch.cat(mu_trajs, dim=0), torch.cat(x_trajs, dim=0), times

def decode_state_lib_traj(model, X_t, model_library_size):
    """Perform decoding at a trajectory snapshot
    """
    if model_library_size:
        mu_t_hat = F.softmax(model.decode(X_t[:, :-1]), dim=1)
        l_t_hat = torch.exp(X_t[:, -1])
        mu_t_hat = mu_t_hat * l_t_hat.unsqueeze(-1)
        X_t_hat = model.sample_decoder(X_t[:, :-1], l_t_hat).detach().cpu()
    else:
        mu_t_hat = torch.exp(model.decode(X_t))
        X_t_hat = model.sample_decoder(X_t).detach().cpu()
    return mu_t_hat.detach().cpu(), X_t_hat

    