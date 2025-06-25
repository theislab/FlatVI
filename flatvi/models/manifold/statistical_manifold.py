from typing import Optional

import torch
from torch.distributions import kl_divergence

from stochman.manifold import Manifold
from stochman.curves import BasicCurve, CubicSpline

from scCFM.models.manifold.geodesic import sc_geodesic_minimizing_energy
from scCFM.models.utils import get_distribution
from scCFM.models.manifold.utils import nb_kl


class scStatisticalManifold(Manifold):
    def curve_energy(self, curve: torch.Tensor):
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
        kl = self._decode_and_kl(curve)
        return torch.sum(kl.view(kl.shape[0], -1), dim=1) * (2 * (dt ** -1))

    def curve_length(self, curve: torch.Tensor):
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
        kl = self._decode_and_kl(curve)
        return torch.sqrt(2 * torch.sum(kl.view(kl.shape[0], -1), dim=1))

    def connecting_geodesic(self, 
                            p0, 
                            p1, 
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
        
        # Train the geodesic spline 
        losses = sc_geodesic_minimizing_energy(curve, 
                                                self, 
                                                max_iter=max_iter, 
                                                eval_grid=eval_grid,
                                                lr=lr, 
                                                minimize_energy=minimize_energy, 
                                                return_losses=return_losses)
        return curve, losses

    def metric(self, z, decoder_outputs):    
        """Metric tensor

        Args:
            z (torch.tensor): latent space point to evaluate metric on 
            decoder_outputs (dict): parameter decoder outputs

        Raises:
            NotImplementedError: only Negative Binomial is implemented

        Returns:
            torch.tensor: metric tensor at point z
        """
        # Compute the value of the Fisher matrix
        if self.model.likelihood == "nb":
            nb_fisher = self.model.theta.unsqueeze(0) / \
                (decoder_outputs["mu"] * (self.model.theta.unsqueeze(0) + decoder_outputs["mu"]))  # B x D
        else:
            raise NotImplementedError
        
        # Use the canonical basis vectors to select separate columns of the decoder (speed reasons)
        basis =  torch.eye(z.shape[1])
        basis = basis.unsqueeze(0).expand(z.shape[0], z.shape[1], z.shape[1])  # B x d x d
        
        # Compute the statistical manifold metric tensor 
        jac = []
        for i in range(z.shape[1]):
            _, jac_partial = torch.func.jvp(self.model.decode,
                                            (z,), 
                                            (basis[:, :, i],))  # B x D
            jac.append(jac_partial)

        jac = torch.stack(jac, dim=-1)  # B x D x d
        return torch.einsum("bij,bik->bjk", jac, jac * nb_fisher.unsqueeze(-1))
    
    def _decode_and_kl(self, curve):
        """Decode the curve 

        Args:
            curve (torch.tensor): a tensor containing the points of a batch of curves.

        Returns:
            float: Kullback-Leibler divergence between consecutive points on the curve 
        """
        # Decode the points on the curve 
        decoder_outputs1 = self.model.decode(curve[:, :-1, :])
        decoder_outputs2 = self.model.decode(curve[:, 1:, :])
        
        if self.model.likelihood == "gaussian":
            dist1 = get_distribution(self.model._preprocess_decoder_output(decoder_outputs1), 
                                     self.model.log_sigma, 
                                     likelihood = self.model.likelihood)
            dist2 = get_distribution(self.model._preprocess_decoder_output(decoder_outputs2), 
                                     self.model.log_sigma,
                                     likelihood = self.model.likelihood)
            kl = kl_divergence(dist1, dist2)
            
        elif self.model.likelihood == "nb" or self.model.likelihood == "zinb":
            dist1 = get_distribution(self.model._preprocess_decoder_output(decoder_outputs1), 
                                     self.model.theta, 
                                     likelihood = self.model.likelihood)
            dist2 = get_distribution(self.model._preprocess_decoder_output(decoder_outputs2), 
                                     self.model.theta, 
                                     likelihood = self.model.likelihood)
            kl = nb_kl(dist1, dist2)  
        
        return kl
