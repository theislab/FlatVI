import torch
import torch.nn.functional as F

def mse_loss(recon_x, x, reduction=None):
    mse_loss = torch.nn.functional.mse_loss(recon_x["mu"], x, reduction=reduction)
    return mse_loss

def nb_loss(mu, theta, x, eps=1e-08, reduction=None):
    if len(theta.shape()) == 1:
        theta = theta.view(1, theta.size(0))

    # negative binomial loss parametrized by mean and dispersion  
    log_theta_mu_eps = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    return res

def zinb_loss(mu, theta, pi, x, eps=1e-08, reduction=None):
    if len(theta.shape()) == 1:
        theta = theta.view(1, theta.size(0))  

    # Ensure positivity pi
    softplus_pi = F.softplus(-pi)  #  uses log(sigmoid(x)) = -softplus(-x)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    return res
