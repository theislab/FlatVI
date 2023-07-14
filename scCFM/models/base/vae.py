from scPDETraversal.modules.mlp import MLP
from scPDETraversal.modules.losses import mse_loss, nb_loss, zinb_loss
import torch
import torch.nn as nn


class VAE(torch.nn.Module):
    def __init__(self, 
                 in_dim, 
                 hidden_dims, 
                 batch_norm, 
                 dropout, 
                 dropout_p, 
                 activation=torch.nn.ReLU, 
                 likelihood="nb"):
        
        super(VAE, self).__init__()

        # Attributes
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.likelihood = likelihood

        # Encoder
        self.encoder_layers = MLP(hidden_dims=[in_dim, *hidden_dims[:-1]],
                                    batch_norm=batch_norm, 
                                    dropout=dropout, 
                                    dropout_p=dropout_p, 
                                    activation=activation)

        # Latent space
        self.latent_dim = hidden_dims[-1]
        self.mu = nn.Linear(hidden_dims[-2], self.latent_dim)
        self.logvar = nn.Linear(hidden_dims[-2], self.latent_dim)

        # Decoder
        self.decoder_layers = MLP(hidden_dims=[*hidden_dims[::-1]],
                                    batch_norm=batch_norm, 
                                    dropout=dropout, 
                                    dropout_p=dropout_p, 
                                    activation=activation)
        
        if likelihood=="gaussian":
            self.theta = None
            self.decoder_mu = torch.nn.Linear(hidden_dims[0], self.in_dim)
            
        elif likelihood=="nb" :
            self.theta = torch.nn.Parameter(torch.randn(self.in_dim))
            self.decoder_mu = torch.nn.Sequential(*[torch.nn.Linear(hidden_dims[0], self.in_dim), 
                                                  torch.nn.Softmax(dim=-1)])
            
        elif likelihood=="zinb": 
            self.theta = torch.nn.Parameter(torch.randn(self.in_dim))
            self.decoder_mu = torch.nn.Sequential(*[torch.nn.Linear(hidden_dims[0], self.in_dim), 
                                                  torch.nn.Softmax(dim=-1)])
            self.decoder_rho = torch.nn.Linear(hidden_dims[0], self.in_dim)
        
        else:
            raise NotImplementedError
        
    def encode(self, x):
        h = self.encoder_layers(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, library_size):
        h = self.decoder_layers(z)
        
        if self.likelihood == "gaussian":
            return self.decoder_mu(h)
        
        elif self.likelihood=="nb":
            return library_size.unsqueeze(-1)*self.decoder_mu(h)
        
        elif self.likelihood=="zinb":
            return library_size.unsqueeze(-1)*self.decoder_mu(h), self.decoder_rho(h)
        
        else:
            raise NotImplementedError
        
    def forward(self, batch):
        x, library_size = batch["x"], batch["library_size"]
        x_log = torch.log(1+x)
        mu, logvar = self.encode(x_log)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, library_size), mu, logvar

    def generate_samples(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        return self.decode(z)

    def reconstruction_loss(self, recon_x, x, rho=None, likelihood='gaussian'):
        if likelihood == 'gaussian':
            recon_loss = mse_loss(recon_x, 
                                  x, 
                                  reduction='mean')
            
        elif likelihood == 'nb':
            recon_loss = nb_loss(recon_x, 
                                 torch.exp(self.theta), 
                                 x, 
                                 reduction="mean")
            
        elif likelihood == 'zinb':
            recon_loss = zinb_loss(recon_x, 
                                 torch.exp(self.theta), 
                                 rho,
                                 x, 
                                 reduction="mean")
            
        else:
            raise NotImplementedError
        
        return recon_loss

    def kl_divergence(self, mu, logvar):
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_div

    def loss_function(self, recon_x, x, mu, logvar, rho=None, likelihood='gaussian', beta=1.0):
        recon_loss = self.reconstruction_loss(recon_x, x, rho, likelihood)
        kl_div = self.kl_divergence(mu, logvar)
        loss = recon_loss + beta * kl_div
        return loss
    
if __name__=="__main__":
    vae = VAE(in_dim=2000, 
                 hidden_dims=[256,256, 256], 
                 batch_norm=True, 
                 dropout=False, 
                 dropout_p=0, 
                 activation=torch.nn.ReLU)
    print(vae)
    