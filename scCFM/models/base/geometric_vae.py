import torch
import torch.nn.functional as F

from scCFM.models.base.vae import AE
    
    
class GeometricNBVAE(AE):
    def __init__(self, manifold, vae_kwargs):
        super(GeometricNBVAE, self).__init__(**vae_kwargs)
        
        assert vae_kwargs["likelihood"] == "nb" or vae_kwargs["likelihood"] == "zinb"
        
        self.manifold = manifold
        
    def mean_function(self, z):
        h = self.decoder_layers(z)
        mu = self.decoder_mu(h)
        mu = F.softmax(mu, dim=-1)
        
        library_size = torch.exp(self.decoder_lib(h))
        library_size = library_size.unsqueeze(-1)
        
        mu = mu * library_size
        return mu 