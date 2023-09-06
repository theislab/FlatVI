import torch
from scvi.distributions import NegativeBinomial

def nb_kl(p: NegativeBinomial, q: NegativeBinomial):
    """KL divergence between two negative binomial distributions 

    Args:
        p (NegativeBinomial): distribution p
        q (NegativeBinomial): distribution q

    Returns:
        float: result of the kl divergence between two distributions
    """
    assert torch.all(p.theta == q.theta)
    theta = p.theta
    mu_p = p.mu
    mu_q = q.mu

    kl = theta*torch.log((theta + mu_q)/(theta + mu_p))
    kl += torch.log((mu_p*(mu_q + theta))/(mu_q*(mu_p+theta)))*mu_p
    return kl
