import torch 
from scCFM.models.utils import get_distribution
from scCFM.models.manifold.utils import nb_kl
# from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef

# corrcoef = SpearmanCorrCoef()

def get_condition_number(J):
    """Condition number

    Args:
        J (torch.tensor): metric tensor of dimension BxDxd

    Returns:
        torch.tensor: the condition number score as the ratio between the maximum and minimum evalues 
    """
    # Eigenvalues 
    S = torch.svd(J).S
    
    # Largest over lower eigenvalue
    scores = S.max(1).values/S.min(1).values
    return scores 
        
def get_variance(J):
    """Variance of the metric tensor 

    Args:
        J (torch.tensor): metric tensor of dimension BxDxd

    Returns:
        torch.tensor: variance score for each observation in the batch 
    """
    # Get mean metric tensor
    J_mean = torch.mean(J, dim=0, keepdim=True)
    # B^-1.dot(A)
    A = torch.inverse(J_mean)@J
    scores = torch.sum(torch.log(torch.svd(A).S)**2, dim=1)
    return scores

def get_magnification_factor(J):
    """Magnification factor as determinant of the metric tensor

    Args:
        J (torch.tensor): metric tensor of dimension BxDxd

    Returns:
        torch.tensor: magnification factor as the squared root of the determinant
    """
    return torch.sqrt(torch.linalg.det(J))

def get_euclidean_to_kl_ratio(z_batch, decoded_outputs, model):
    """Compare latent distances witht the kl distances between observations 

    Args:
        z_batch (torch.tensor): batch of latent observations 
        decoded_outputs (dict): dictionary of outputs of the decoder 
        model (torch.nn.model): autoencoder model

    Raises:
        NotImplementedError: the model is only implemented for negative binomial 

    Returns:
        torch.tensor: absolute difference between the latent Euclidean and data kl losses
    """
    bs, _ = z_batch.size()
    # permute observations 
    random_perm = torch.randperm(bs)
    z_permuted = z_batch[random_perm]
    decoded_outputs_permuted = {key: value[random_perm] for key, value in decoded_outputs.items()}
    # Euclidean latent 
    euclidean_dist_latent = torch.sum((z_permuted - z_batch)**2, dim=-1).detach().cpu()
    # KL true 
    if model.likelihood == "nb":
        distr_batch = get_distribution(decoded_outputs, model.theta, likelihood=model.likelihood)
        distr_perm = get_distribution(decoded_outputs_permuted, model.theta, likelihood=model.likelihood)
        kl_dist = nb_kl(distr_batch, distr_perm).sum(-1).detach().cpu()
    else:
        raise NotImplementedError
    return torch.mean(torch.abs(euclidean_dist_latent - kl_dist))
    
def compute_all_metrics(J, z_batch, decoded_outputs, model, average=True):
    model.train()
    cn = get_condition_number(J)
    var = get_variance(J)
    mf = get_magnification_factor(J)
    eu_kl_dist = get_euclidean_to_kl_ratio(z_batch, decoded_outputs, model)
    
    if average:
        dict_metrics = {"condition_number": cn.mean(),
                        "variance": var.mean(),
                        "magnification_factor": mf.mean(), 
                        "eu_kl_dist": eu_kl_dist.mean()}
    else:
        dict_metrics = {"condition_number": cn,
                "variance": var,
                "magnification_factor": mf, 
                "eu_kl_dist": eu_kl_dist}
    model.eval()
    return dict_metrics
