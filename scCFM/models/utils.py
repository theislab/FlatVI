import torch
from pytorch_lightning.loggers import WandbLogger
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from torch.distributions import Normal 

def pad_t_like_x(t, x):
    """Pad tensor t as x

    Args:
        t (torch.tensor): tensor to pad
        x (torch.tensor): tensor whose dimensions are used for the padding

    Returns:
        torch.tensor: tensor t after padding
    """
    if isinstance(t, float):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

def get_distribution(decoder_output, log_var, likelihood: str = "nb"):
    """Given decoded outputs, decode distributions 

    Args:
        decoder_output (dict): dictionary with the outputs from the decoder
        log_var (torch.tensor): either log variance of a Gaussian or the log over-dispersion
        likelihood (str, optional): type of likelihood. Defaults to "nb".

    Raises:
        NotImplementedError: if the selected likelihood is not Gaussian or (zero-inflated) Negative Binomial

    Returns:
        torch.distributions: fitted distribution 
    """
    if likelihood == "gaussian":
        mu = decoder_output["mu"]
        distr = Normal(loc=mu, theta=torch.sqrt(torch.exp(log_var)).unsqueeze(0).expand(mu.shape[0], mu.shape[1]))

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

def jacobian_decoder_jvp_parallel(func, inputs, v=None, create_graph=True):
    """Fast Jacobian computation for batched input 

    Args:
        func (torch.nn.module): decoder function to apply
        inputs (torch.tensor): point where Jacobian is evaluated
        v (torch.tensor, optional): tangent vector where Jacobian is evaluated. Defaults to None.
        create_graph (bool, optional): whether to create graph for gradient. Defaults to True.

    Returns:
        torch.tensor: Jacobian of func at point inputs with respect to tangent vector v
    """
    batch_size, z_dim = inputs.size()
    if v is None:
        v = torch.eye(z_dim).unsqueeze(0).repeat(batch_size, 1, 1).view(-1, z_dim).to(inputs)
    inputs = inputs.repeat(1, z_dim).view(-1, z_dim)
    jac = (
        torch.autograd.functional.jvp(
            func, inputs, v=v, create_graph=create_graph
        )[1].view(batch_size, z_dim, -1).permute(0, 2, 1)
    )
    return jac

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))
    