import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
import pytorch_lightning as pl

from scCFM.models.base.mlp import MLP
from scCFM.models.base.losses import mse_loss, nb_loss, zinb_loss


class BaseAutoencoder(pl.LightningModule):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        batch_norm,
        dropout,
        dropout_p,
        activation=torch.nn.ReLU,
        likelihood="nb",
    ):
        super(BaseAutoencoder, self).__init__()

        # Attributes
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.likelihood = likelihood

        # Encoder
        self.encoder_layers = MLP(
            hidden_dims=[in_dim, *hidden_dims[:-1]],
            batch_norm=batch_norm,
            dropout=dropout,
            dropout_p=dropout_p,
            activation=activation,
        )

        # Decoder
        self.decoder_layers = MLP(
            hidden_dims=[*hidden_dims[::-1]],
            batch_norm=batch_norm,
            dropout=dropout,
            dropout_p=dropout_p,
            activation=activation,
        )

        if likelihood == "gaussian":
            self.theta = None
            self.decoder_mu = torch.nn.Linear(hidden_dims[-1], self.in_dim)

        elif likelihood == "nb":
            self.theta = torch.nn.Parameter(torch.randn(self.in_dim))
            self.decoder_mu = torch.nn.Linear(hidden_dims[0], self.in_dim)

        elif likelihood == "zinb":
            self.theta = torch.nn.Parameter(torch.randn(self.in_dim))
            self.decoder_mu = torch.nn.Linear(hidden_dims[0], self.in_dim)
            self.decoder_rho = torch.nn.Linear(hidden_dims[0], self.in_dim)

        else:
            raise NotImplementedError

    def encode(self, x):
        pass

    def decode(self, z, library_size):
        h = self.decoder_layers(z)

        if self.likelihood == "gaussian":
            mu = self.decoder_mu(h)
            return dict(mu=mu)

        elif self.likelihood == "nb":
            mu = self.decoder_mu(h)
            mu = F.softmax(mu, dim=-1)
            mu = mu*library_size.unsqueeze(-1)
            return dict(mu=mu)

        elif self.likelihood == "zinb":
            mu = self.decoder_mu(h)
            mu = F.softmax(mu, dim=-1)
            mu = mu*library_size.unsqueeze(-1)
            rho = self.decoder_rho(h)
            return dict(mu=mu, rho=rho)

        else:
            raise NotImplementedError

    def forward(self, batch):
        pass

    def reconstruction_loss(self, x, decoder_output):
        if self.likelihood == "gaussian":
            recon_loss = mse_loss(decoder_output, x, reduction="sum")

        elif self.likelihood == "nb":
            mu = decoder_output["mu"]
            distr = NegativeBinomial(mu=mu, theta=torch.exp(self.theta))
            recon_loss = -distr.log_prob(x).sum()

        elif self.likelihood == "zinb":
            mu, rho = decoder_output["mu"], decoder_output["rho"]
            distr = ZeroInflatedNegativeBinomial(
                mu=mu, theta=torch.exp(self.theta), zi_logits=rho
            )
            recon_loss = -distr.log_prob(x).sum()

        else:
            raise NotImplementedError
        return recon_loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


class Autoencoder(BaseAutoencoder):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        batch_norm,
        dropout,
        dropout_p,
        activation=torch.nn.ELU,
        likelihood="nb",
    ):
        super(Autoencoder, self).__init__(
            in_dim,
            hidden_dims,
            batch_norm,
            dropout,
            dropout_p,
            activation,
            likelihood,
        )
        
        self.latent_layer = torch.nn.Linear(hidden_dims[-1], self.in_dim)
    
    def encode(self, x):
        h = self.encoder_layers(x)
        return self.latent_layer(h)
    
    def forward(self, batch):
        x = batch["X"]
        library_size = x.sum(1)
        x_log = torch.log(1 + x)
        latent_representation = self.encode(x_log)
        return self.decode(latent_representation, library_size)

    def training_step(self, batch, batch_idx):
        x = batch["X"]
        decoder_output = self.forward(batch)
        recon_loss = self.reconstruction_loss(x, decoder_output)
        loss = recon_loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["X"]
        decoder_output = self.forward(batch)
        recon_loss = self.reconstruction_loss(x, decoder_output)
        loss = recon_loss
        self.log("val_loss", loss, prog_bar=True)

class VAE(BaseAutoencoder):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        batch_norm,
        dropout,
        dropout_p,
        activation=torch.nn.ReLU,
        likelihood="nb",
    ):
        super(VAE, self).__init__(
            in_dim,
            hidden_dims,
            batch_norm,
            dropout,
            dropout_p,
            activation,
            likelihood,
        )

        # Latent space
        self.latent_dim = hidden_dims[-1]
        self.mu = nn.Linear(hidden_dims[-2], self.latent_dim)
        self.logvar = nn.Linear(hidden_dims[-2], self.latent_dim)
    
    def encode(self, x):
        h = self.encoder_layers(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, batch):
        x = batch["X"]
        library_size = x.sum(1)
        x_log = torch.log(1 + x)
        mu, logvar = self.encode(x_log)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, library_size), mu, logvar

    def kl_divergence(self, mu, logvar):
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_div

    def training_step(self, batch, batch_idx):
        x = batch["X"]
        decoder_output, mu, logvar = self.forward(batch)
        recon_loss = self.reconstruction_loss(x, decoder_output)
        kl_div = self.kl_divergence(mu, logvar)
        loss = recon_loss + kl_div
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["X"]
        decoder_output, mu, logvar = self.forward(batch)
        recon_loss = self.reconstruction_loss(x, decoder_output)
        kl_div = self.kl_divergence(mu, logvar)
        loss = recon_loss + kl_div
        self.log("val/loss", loss, prog_bar=True)
        