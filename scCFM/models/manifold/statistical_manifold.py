from typing import Optional, Tuple

import torch
from torch.distributions import kl_divergence
from torch.autograd.functional import jvp as jac

from stochman.manifold import Manifold
from stochman.curves import BasicCurve, CubicSpline

from scCFM.models.manifold.geodesic import geodesic_minimizing_energy_with_library
from scCFM.models.utils import get_distribution
from scCFM.models.manifold.utils import nb_kl


class scStatisticalManifold(Manifold):
    def __init__(self, model: torch.nn.Module) -> None:
        """
        Class constructor:

        Arguments:
        - model: a torch VAE model with decode function.
        """
        super().__init__()
        # Model must be an autoencoder model 
        self.model = model
        self.model.eval()
        assert "decode" in dir(model)

    def curve_energy(self, curve: torch.Tensor, library_size: float, likelihood: str = "nb"):
        """
        Returns the curve energy in the statistical manifold, according
        to the pullback of the Fisher-Rao metric.

        See Pulling Back Information Geometry, Eq. (11) or Proposition A.2.

        Input:
            curve: a tensor containing the points of a batch of curves.
                   We expect `curve` to be a b x n_t x d tensor, where b is
                   the number of curves to evaluate, n_t is the number of points
                   in the curve itself, and d is the dimension of the latent space.
                   if the curve only has two shapes, a batch shape is unsqueezed.

                   The reasoning behind this specific set-up: it is exactly what
                   the DiscretizedManifold is expecting when fitting. See
                   discretized_manifold.py for more details.

        Output:
            length: a tensor of shape (b,) containing the lengths of each
                    of the curves.
        """
        if len(curve.shape) == 2:
            curve = curve.unsqueeze(0)
        
        # Compute pairwise distances between subsequent points 
        dt = (curve[:, :-1] - curve[:, 1:]).pow(2).sum(dim=-1).sqrt().squeeze(1)  # b
        
        # Decode the points on the curve
        decoder_outputs1 = self.model.decode(curve[:, :-1, :], library_size=library_size)
        decoder_outputs2 = self.model.decode(curve[:, 1:, :], library_size=library_size)
        
        if likelihood == "gaussian":
            dist1 = get_distribution(decoder_outputs1, self.model.log_sigma, likelihood = likelihood)
            dist2 = get_distribution(decoder_outputs2, self.model.log_sigma, likelihood = likelihood)
            kl = kl_divergence(dist1, dist2)
            
        elif likelihood == "nb" or likelihood == "zinb":
            dist1 = get_distribution(decoder_outputs1, self.model.theta, likelihood = likelihood)
            dist2 = get_distribution(decoder_outputs2, self.model.theta, likelihood = likelihood)
            kl = nb_kl(dist1, dist2)

        return torch.sum(kl.view(kl.shape[0], -1), dim=1) * (2 * (dt ** -1))

    def curve_length(self, curve: torch.Tensor, library_size: float, likelihood: str = "nb"):
        """
        Returns the curve length in the statistical manifold, according
        to the pullback of the Fisher-Rao metric.

        See Pulling Back Information Geometry, Eq. (10) or Proposition A.2.

        Input:
            curve: a tensor containing the points of a batch of curves.
                   We expect `curve` to be a b x n_t x d tensor, where b is
                   the number of curves to evaluate, n_t is the number of points
                   in the curve itself, and d is the dimension of the latent space.
                   if the curve only has two shapes, a batch shape is unsqueezed.

                   The reasoning behind this specific set-up: it is exactly what
                   the DiscretizedManifold is expecting when fitting. See
                   discretized_manifold.py for more details.

        Output:
            length: a tensor of shape (b,) containing the lengths of each
                    of the curves.
        """
        if len(curve.shape) == 2:
            curve = curve.unsqueeze(0)

        # Decode the points on the curve 
        decoder_outputs1 = self.model.decode(curve[:, :-1, :], library_size=library_size)
        decoder_outputs2 = self.model.decode(curve[:, 1:, :], library_size=library_size)
        
        if likelihood == "gaussian":
            dist1 = get_distribution(decoder_outputs1, self.model.log_sigma, likelihood = likelihood)
            dist2 = get_distribution(decoder_outputs2, self.model.log_sigma, likelihood = likelihood)
            kl = kl_divergence(dist1, dist2)
            
        elif likelihood == "nb" or likelihood == "zinb":
            dist1 = get_distribution(decoder_outputs1, self.model.theta, likelihood = likelihood)
            dist2 = get_distribution(decoder_outputs2, self.model.theta, likelihood = likelihood)
            kl = nb_kl(dist1, dist2)

        return torch.sqrt(2 * torch.sum(kl.view(kl.shape[0], -1), dim=1))

    def connecting_geodesic(self, 
                            p0, 
                            p1, 
                            library_size,
                            init_curve: Optional[BasicCurve] = None, 
                            max_iter=500, 
                            eval_grid=100, 
                            lr=1e-2, 
                            minimize_energy=True, 
                            return_losses=True):
        """
        Compute geodesic connecting two points.

        Args:
            p0: a torch Tensor representing the initial point of the requested geodesic.
            p1: a torch Tensor representing the end point of the requested geodesic.
            init_curve: a curve representing an initial guess of the requested geodesic.
                If the end-points of the initial curve do not correspond to p0 and p1,
                then the curve is modified accordingly. If None then the default constructor
                of the chosen curve family is applied.
        """
         # Decouple curve and library size
        if init_curve is None:
            curve = CubicSpline(p0, p1)
        else:
            curve = init_curve
            curve.begin = p0
            curve.end = p1

        losses = geodesic_minimizing_energy_with_library(curve, 
                                                            self, 
                                                            library_size, 
                                                            max_iter=max_iter, 
                                                            eval_grid=eval_grid,
                                                            lr=lr, 
                                                            minimize_energy=minimize_energy, 
                                                            return_losses=return_losses)
        return curve, losses

    def metric(self, p):    
        # Compute decoder's Jacobian 
        mu, jv = jac(func=self.model.decode,
                             inputs=p,
                             create_graph=True)
        
        if self.likelihood == "nb":
            # Compute the Fisher matrix 
            nb_fisher = self.model.theta / (mu * (self.model.theta + mu))
            
        
        
                             
                              
                              
                             