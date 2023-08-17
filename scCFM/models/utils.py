import torch
from pytorch_lightning.loggers import WandbLogger
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from torch.distributions import Normal, kl_divergence 

def pad_t_like_x(t, x):
    if isinstance(t, float):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

def get_wandb_logger(loggers):
    """Gets the wandb logger if it is the list of loggers otherwise returns None."""
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            wandb_logger = logger
    return wandb_logger

def get_distribution(decoder_output, log_var, likelihood: str = "nb"):
    if likelihood == "gaussian":
        mu = decoder_output["mu"]
        distr = Normal(loc=mu, theta=torch.sqrt(torch.exp(log_var)).unsqueeze(-1)*torch.eye(mu.shape[1]))

    elif likelihood == "nb":
        mu = decoder_output["mu"]
        distr = NegativeBinomial(mu=mu, theta=torch.exp(log_var))

    elif likelihood == "zinb":
        mu, rho = decoder_output["mu"], decoder_output["rho"]
        distr = ZeroInflatedNegativeBinomial(mu=mu, theta=torch.exp(log_var), zi_logits=rho)
        
    else:
        raise NotImplementedError
    
    return distr

def three_d_to_two_d(t):
    return t.view(-1, t.shape[-1])

def two_d_to_three_d(t, time_steps):
    return t.view(-1, time_steps, t.shape[-1])
