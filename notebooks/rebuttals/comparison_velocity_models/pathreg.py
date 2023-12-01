import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import scanpy as sc
import seaborn as sns
from torchdiffeq import odeint
import copy
from L0_regularization import L0Dense

import sys 
import os

class initial_position(nn.Module): 
    def __init__(self, dim, nhidden):
        super(initial_position, self).__init__()
        
    def forward(self, x): 
        
        x0 = torch.mean(x,axis=0)
        for g in range (x.shape[2]):
            zscore = (x[...,g] - x[...,g].mean()) / torch.sqrt(x[...,g].var())
            zscore = torch.where(torch.isnan(zscore), torch.zeros_like(zscore), zscore)
            x0[:,g] = torch.mean(x[...,g][zscore<3])
        return x0  

class ODEBlock(nn.Module):

    def __init__(self, odefunc, dim):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.dim = dim
        self.args = dict(tol=1e-3,visualize=True,niters=5000,ntimestamps =5,lr=0.01,gpu=0,method = 'dopri5', n_samples=1)
        
    def set_times(self,times):
        self.integration_times = times
        
    def forward(self, x):
        integrated = odeint(self.odefunc, 
                            x, 
                            self.integration_times, 
                            rtol=self.args['tol'],
                            atol=self.args['tol'], 
                            method = self.args['method'])
        return integrated

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class L0_MLP(nn.Module):
    def __init__(self, input_dim, layer_dims=(100, 100), N=50000, beta_ema=0.999,
                 weight_decay=0., lambas=(1., 1., 1.), local_rep=False, temperature=2./3.):
        super(L0_MLP, self).__init__()
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.N = N
        self.beta_ema = beta_ema
        self.weight_decay = self.N * weight_decay
        self.lambas = lambas

        layers = []
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = self.input_dim if i == 0 else self.layer_dims[i - 1]
            droprate_init, lamba = 0.2 if i == 0 else 0.5, lambas[i] if len(lambas) > 1 else lambas[0]
            if i<len(self.layer_dims)-2:
                layers += [L0Dense(inp_dim, dimh, droprate_init=droprate_init, weight_decay=self.weight_decay,
                               lamba=lamba, local_rep=local_rep, temperature=temperature)]
                layers += [nn.ELU(alpha=1, inplace=False)]
            else:
                layers += [L0Dense(inp_dim, dimh, droprate_init=droprate_init, weight_decay=self.weight_decay,
                               lamba=lamba, local_rep=local_rep, temperature=temperature)]
        self.output = nn.Sequential(*layers)

        self.layers = []
        for m in self.modules():
            if isinstance(m, L0Dense):
                self.layers.append(m)
        
        self.nfe = 0
            
        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = copy.deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.
        self.nfe = 0

       
    def forward(self, t, x):  
        self.nfe += 1
        return self.output(x)
    
    def regularization(self):
        regularization = 0.
        for layer in self.l0_layers:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.l0_layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = copy.deepcopy(list(p.data for p in self.parameters()))
        return params
    
def PathReg(model):
    for i, layer in enumerate(model[0].odefunc.layers):
        if i ==0:
            WM = torch.abs(layer.sample_weights_ones())
        else:
            WM = torch.matmul(WM,torch.abs(layer.sample_weights_ones()))
    return torch.mean(torch.abs(WM))

def L1(model):
    for i, layer in enumerate(model[0].odefunc.layers):
        if i ==0:
            WM = torch.abs(layer.weights)
        else:
            WM = torch.matmul(WM,torch.abs(layer.weights))
    return torch.mean(torch.abs(WM))
