import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from torchdyn.core import NeuralODE


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
        start = obs[0][:n]
        ts = len(obs)
    else:
        batch_size, ts, dim = obs.shape
        start = obs[:n, start_time, :]
        obs = obs.reshape(-1, dim).detach().cpu().numpy()
        tts = np.tile(np.arange(ts), batch_size)

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
        start = obs[0][:n]
        ts = len(obs)
    else:
        batch_size, ts, dim = obs.shape
        start = obs[:n, start_time, :]
        obs = obs.reshape(-1, dim).detach().cpu().numpy()
        tts = np.tile(np.arange(ts), batch_size)

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
        