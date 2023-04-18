import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scprep
import torch


def plot_scatter(obs, model, title="fig", wandb_logger=None):
    """Scatterplot of observations
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    batch_size, ts, dim = obs.shape
    # Remove time dimension 
    obs = obs.reshape(-1, dim).detach().cpu().numpy()
    ts = np.tile(np.arange(ts), batch_size)  # Repeat time vector for batch size times 
    scprep.plot.scatter2d(obs, c=ts, ax=ax)
    if not os.path.exists("figs"):
        os.mkdir("figs")
    plt.savefig(f"figs/{title}.png")
    if wandb_logger:
        wandb_logger.log_image(key=title, images=[f"figs/{title}.png"])
    plt.close()


def plot_scatter_and_flow(obs, model, title="stream", wandb_logger=None):
    """Plot path and scatter plot (not sure about the latter - must check)
    """
    batch_size, ts, dim = obs.shape
    device = obs.device
    # Remove time dimension
    obs = obs.reshape(-1, dim).detach().cpu().numpy() # (TB)xF
    # Boundary plot
    diff = obs.max() - obs.min()
    wmin = obs.min() - diff * 0.1
    wmax = obs.max() + diff * 0.1
    points = 50j
    points_real = 50
    Y, X, T = np.mgrid[wmin:wmax:points, wmin:wmax:points, 0 : ts - 1 : 7j]
    gridpoints = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1), requires_grad=True
    ).type(torch.float32)
    times = torch.tensor(T.flatten(), requires_grad=True).type(torch.float32)[:, None]
    # Model the output of the vector field prediction 
    out = model(times.to(device), gridpoints.to(device))
    out = out.reshape([points_real, points_real, 7, dim])
    out = out.cpu().detach().numpy()
    # Stream over time
    fig, axes = plt.subplots(1, 7, figsize=(20, 4), sharey=True)
    axes = axes.flatten()
    tts = np.tile(np.arange(ts), batch_size)
    for i in range(7):
        scprep.plot.scatter2d(obs, c=tts, ax=axes[i])
        axes[i].streamplot(
            X[:, :, 0],
            Y[:, :, 0],
            out[:, :, i, 0],
            out[:, :, i, 1],
            color=np.sum(out[:, :, i] ** 2, axis=-1),
        )
        axes[i].set_title(f"t = {np.linspace(0,ts-1,7)[i]:0.2f}")
    if not os.path.exists("figs"):
        os.mkdir("figs")
    plt.savefig(f"figs/{title}.png")
    plt.close()
    if wandb_logger:
        wandb_logger.log_image(key="flow", images=[f"figs/{title}.png"])


def store_trajectories(obs: Union[torch.Tensor, list], model, title="trajs", start_time=0):
    n = 2000
    if isinstance(obs, list):
        data, labels = [], []
        for t, xi in enumerate(obs):
            xi = xi.detach().cpu().numpy()
            data.append(xi)
            labels.append(t * np.ones(xi.shape[0]))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        scprep.plot.scatter2d(data, c=labels)
        start = obs[0][:n]
        ts = len(obs)
    else:
        batch_size, ts, dim = obs.shape
        start = obs[:n, start_time, :]
        obs = obs.reshape(-1, dim).detach().cpu().numpy()
        tts = np.tile(np.arange(ts), batch_size)
    from torchdyn.core import NeuralODE

    with torch.no_grad():
        node = NeuralODE(model)
        # For consistency with DSB
        traj = node.trajectory(start, t_span=torch.linspace(0, ts - 1, 20 * (ts - 1)))
        traj = traj.cpu().detach().numpy()
        if not os.path.exists("figs"):
            os.mkdir("figs")
        np.save(f"figs/{title}.npy", traj)


def plot_paths(
    obs: Union[torch.Tensor, list], model, title="paths", start_time=0, wandb_logger=None
):
    plt.figure(figsize=(6, 6))
    n = 200
    if isinstance(obs, list):
        data, labels = [], []
        for t, xi in enumerate(obs):
            xi = xi.detach().cpu().numpy()
            data.append(xi)
            labels.append(t * np.ones(xi.shape[0]))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        scprep.plot.scatter2d(data, c=labels)
        start = obs[0][:n]
        ts = len(obs)
    else:
        batch_size, ts, dim = obs.shape
        start = obs[:n, start_time, :]
        obs = obs.reshape(-1, dim).detach().cpu().numpy()
        tts = np.tile(np.arange(ts), batch_size)
        scprep.plot.scatter2d(obs, c=tts)
    from torchdyn.core import NeuralODE

    with torch.no_grad():
        node = NeuralODE(model)
        traj = node.trajectory(start, t_span=torch.linspace(0, ts - 1, max(20 * ts, 100)))
        traj = traj.cpu().detach().numpy()
    # plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.3, alpha=0.2, c="black", label="Flow")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=6, alpha=1, c="purple", marker="x")
    # plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    if not os.path.exists("figs"):
        os.mkdir("figs")
    plt.savefig(f"figs/{title}.png")
    plt.close()
    if wandb_logger:
        wandb_logger.log_image(key="paths", images=[f"figs/{title}.png"])
        