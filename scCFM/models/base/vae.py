import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal, kl_divergence
import pytorch_lightning as pl

from scCFM.models.base.mlp import MLP
from scCFM.models.utils import get_distribution

class BaseAutoencoder(pl.LightningModule):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        batch_norm,
        dropout,
        dropout_p,
        activation=torch.nn.ELU,
        likelihood="nb", 
        learning_rate=1e-4, 
        model_library_size=True
        ):
        """Base autoencoder model

        Args:
            in_dim (int): dimension of the samples
            hidden_dims (list): list with the hidden dimensions of encoder and decoder 
            batch_norm (bool): whether to apply batch normalization 
            dropout (boool): whether to apply dropout or not 
            dropout_p (float): the probability of dropout
            activation (torch.nn, optional): the activation function to use in the neural layers. Defaults to torch.nn.ELU.
            likelihood (str), optional): the likelihood of the decoder. Defaults to "nb".
            learning_rate (float, optional): the learning rate for optimization. Defaults to 1e-4.
            model_library_size (bool, optional): whether to model library size as an additional decoder output. Defaults to True.

        Raises:
            NotImplementedError: in case a likelihood which is not Gaussian, Negative Binomial or Zero Inflated Negative Binomial is selected 
        """
        super(BaseAutoencoder, self).__init__()

        # Attributes
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.activation = activation
        self.likelihood = likelihood
        self.latent_dim = hidden_dims[-1]
        self.learning_rate = learning_rate
        self.model_library_size = float(model_library_size)

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
            self.log_sigma = torch.nn.Parameter(torch.randn(self.in_dim))
            self.decoder_mu = torch.nn.Linear(hidden_dims[0], self.in_dim)

        elif likelihood == "nb":
            self.theta = torch.nn.Parameter(torch.randn(self.in_dim))
            self.decoder_mu_lib = torch.nn.Linear(hidden_dims[0], self.in_dim + model_library_size)

        elif likelihood == "zinb":
            self.theta = torch.nn.Parameter(torch.randn(self.in_dim))
            self.decoder_mu_rho_lib = torch.nn.Linear(hidden_dims[0], self.in_dim * 2 + model_library_size)

        else:
            raise NotImplementedError
        
    def encode(self, x):
        pass

    def decode(self, z):
        """Decoder function

        Args:
            z (torch.tensor): the input latent space

        Returns:
            torch.tensor: output of the decoder
        """
        # Decode z layers
        h = self.decoder_layers(z)
        
        # Different decoder options 
        if self.likelihood == "gaussian":
            return self.decoder_mu(h)

        elif self.likelihood == "nb":
            return self.decoder_mu_lib(h)
                
        elif self.likelihood == "zinb":
            return self.decoder_mu_rho_lib(h)

        else:
            raise NotImplementedError

    def forward(self, batch):
        pass

    def reconstruction_loss(self, x, decoder_output):
        """Reconstruction loss (likelihood)

        Args:
            x (torch.tensor): data point to be reconstructed 
            decoder_output (dict): the output of the decoder 

        Returns:
            torch.tensor: reconstruction loss
        """
        if self.likelihood == "gaussian":
            distr = get_distribution(decoder_output, self.log_sigma, likelihood = self.likelihood)

        elif self.likelihood == "nb" or self.likelihood == "zinb":
            distr = get_distribution(decoder_output, self.theta, likelihood = self.likelihood)

        else:
            raise NotImplementedError
        
        recon_loss = -distr.log_prob(x).sum(-1)
        return recon_loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def _preprocess_decoder_output(self, out):
        """Process output of the decoder 

        Args:
            out (torch.tensor): output of the decoder

        Returns:
            dict: values of the decoder outputs
        """
        if self.likelihood == "nb" or self.likelihood == "zinb":
            # Library size is last dimension of the decoder
            if self.model_library_size:
                if self.likelihood == "nb":
                    mu, library_size = out[:, :-1], out[:, -1] 
                elif self.likelihood == "zinb":
                    mu, rho, library_size = out[:, :self.in_dim], out[:, self.in_dim:-1], out[:, -1]
                
                # Preprocess mu 
                mu = F.softmax(mu, dim=-1)
                library_size = torch.exp(library_size)
                mu = mu * library_size.unsqueeze(-1)            

            else:
                if self.likelihood == "nb":
                    mu = out
                elif self.likelihood == "zinb":
                    mu, rho = out[:, :self.in_dim], out[:, self.in_dim:-1]
                
                # We model the log mean 
                mu = torch.exp(mu) 
        
        else:
            mu = out           

        # Return zinb only if negative binomial
        if self.likelihood == "zinb":
            return dict(mu=mu, rho=rho)
        else:
            return dict(mu=mu)
        
    def training_step(self, batch, batch_idx):
        """Training step

        Args:
            batch (dict): batch containing observations and conditions
            batch_idx (torch.tensor): indices of the observations in the batch

        Returns:
            torch.tensor: loss
        """
        loss = self.step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step

        Args:
            batch (dict): batch containing observations and conditions
            batch_idx (torch.tensor): indices of the observations in the batch

        Returns:
            torch.tensor: loss
        """
        self.step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        """Validation step

        Args:
            batch (dict): batch containing observations and conditions
            batch_idx (torch.tensor): indices of the observations in the batch

        Returns:
            torch.tensor: loss
        """
        self.step(batch, "test")
        
class AE(BaseAutoencoder):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        batch_norm,
        dropout,
        dropout_p,
        activation=torch.nn.ELU,
        likelihood="nb", 
        learning_rate=1e-4,
        model_library_size=False
        ):
        """Standard autoencoder class

        Args:
            in_dim (int): dimension of the samples
            hidden_dims (list): list with the hidden dimensions of encoder and decoder 
            batch_norm (bool): whether to apply batch normalization 
            dropout (boool): whether to apply dropout or not 
            dropout_p (float): the probability of dropout
            activation (torch.nn, optional): the activation function to use in the neural layers. Defaults to torch.nn.ELU.
            likelihood (str), optional): the likelihood of the decoder. Defaults to "nb".
            learning_rate (float, optional): the learning rate for optimization. Defaults to 1e-4.
            model_library_size (bool, optional): whether to model library size as an additional decoder output. Defaults to True.
        """
        super(AE, self).__init__(
            in_dim,
            hidden_dims,
            batch_norm,
            dropout,
            dropout_p,
            activation,
            likelihood, 
            learning_rate, 
            model_library_size
        )

        self.latent_layer = torch.nn.Linear(hidden_dims[-2], self.latent_dim)
    
    def encode(self, x):
        """Encoder function

        Args:
            x (torch.tensor): input of the encoder

        Returns:
            dictionary: dictionary with the encoder output 
        """
        x = torch.log1p(x)  
        h = self.encoder_layers(x)
        z = self.latent_layer(h)
        return dict(z=z)
    
    def forward(self, batch):
        """Encoding and decoding forward pass

        Args:
            batch (dictionary): batch containing observations and conditions

        Returns:
            torch.tensor: decoded input 
        """
        x = batch["X"]
        z = self.encode(x)["z"]
        return self.decode(z)
    
    def step(self, batch, prefix):
        """Step of the trained model 

        Args:
            batch (dict): batch containing observations and conditions
            prefix (str): prefix of the type of pass (training or validation)

        Returns:
            torch.tensor: loss
        """
        x = batch["X"]
        decoder_output = self.forward(batch)
        loss = torch.mean(self.reconstruction_loss(x, self._preprocess_decoder_output(decoder_output)))
        self.log(f"{prefix}/loss", loss, prog_bar=x)
        if prefix == "train":
            return loss

class VAE(BaseAutoencoder):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        batch_norm,
        dropout,
        dropout_p,
        n_epochs_anneal_kl,
        kl_warmup_fraction = 0.5,
        kl_weight = None, 
        activation = torch.nn.ELU,
        likelihood = "nb",
        learning_rate = 1e-4, 
        model_library_size = False
        ):
        """Standard variational autoencoder class

        Args:
            in_dim (int): dimension of the samples
            hidden_dims (list): list with the hidden dimensions of encoder and decoder 
            batch_norm (bool): whether to apply batch normalization 
            dropout (boool): whether to apply dropout or not 
            dropout_p (float): the probability of dropout
            n_epochs_anneal_kl (int): for how many epochs the kl divergence should be annealed
            kl_warmup_fraction (float): fraction of steps for which kl is annealed 
            kl_weight (float): weight for the kl. None when performing annealing
            activation (torch.nn, optional): the activation function to use in the neural layers. Defaults to torch.nn.ELU.
            likelihood (str), optional): the likelihood of the decoder. Defaults to "nb".
            learning_rate (float, optional): the learning rate for optimization. Defaults to 1e-4.
            model_library_size (bool, optional): whether to model library size as an additional decoder output. Defaults to True.
        """
        super(VAE, self).__init__(
            in_dim,
            hidden_dims,
            batch_norm,
            dropout,
            dropout_p,
            activation,
            likelihood,
            learning_rate,
            model_library_size
        )

        # Latent space
        self.mu_logvar = nn.Linear(hidden_dims[-2], self.latent_dim * 2)
        if kl_weight==None:
            self.kl_weight = 0
            self.kl_weight_increase = 1/int(kl_warmup_fraction*n_epochs_anneal_kl)
            self.anneal_kl = True
        else:
            self.kl_weight = kl_weight
            self.anneal_kl = False
        
    def encode(self, x):
        """Encoder function

        Args:
            x (torch.tensor): input of the encoder

        Returns:
            dictionary: dictionary with the encoder output 
        """
        x = torch.log1p(x)  # For numerical stability
        h = self.encoder_layers(x)
        out = self.mu_logvar(h)
        mu, logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)
        return dict(z=z,
                    mu=mu,
                    logvar=logvar)

    def reparameterize(self, mu, logvar):
        """Gaussian reparametrization

        Args:
            mu (torch.tensor): latent mean
            logvar (torch.tensor): latent variance

        Returns:
            torch.tensor: sample from normal distribution parametrized by mu and logvar
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, batch):
        """Encoding and decoding forward pass

        Args:
            batch (dictionary): batch containing observations and conditions

        Returns:
            torch.tensor: decoded input 
        """
        x = batch["X"]      
        z, mu, logvar = self.encode(x).values()
        return self.decode(z), z, mu, logvar

    def kl_divergence(self, mu, logvar):
        """Kullback-Leibler divergence between normal distributions 

        Args:
            mu (torch.tensor): mean of the distribution
            logvar (toerch.tensor): log-variance of the distribution

        Returns:
            torch.tensor: Kullback Leibler divergence 
        """
        p =  Normal(mu, torch.sqrt(torch.exp(0.5 * logvar)))
        q = Normal(torch.zeros_like(mu), torch.ones_like(mu))
        kl_div = kl_divergence(p, q).sum(dim=-1)
        return kl_div
        
    def step(self, batch, prefix):
        """Step of the trained model 

        Args:
            batch (dict): batch containing observations and conditions
            prefix (str): prefix of the type of pass (training or validation)

        Returns:
            torch.tensor: loss
        """
        x = batch["X"]
        decoder_output, _, mu, logvar = self.forward(batch)
        # Raw decoder output must be processed befor reconstruction loss
        recon_loss = self.reconstruction_loss(x, self._preprocess_decoder_output(decoder_output))
        kl_div = self.kl_divergence(mu, logvar)
        kl_weight = min([self.kl_weight, 1])

        loss = torch.mean(recon_loss + kl_weight * kl_div)
        dict_losses = {f"{prefix}/loss": loss,
                       f"{prefix}/kl": kl_div.mean(),
                       f"{prefix}/lik": recon_loss.mean()}
        self.log_dict(dict_losses, prog_bar=True)
        if prefix == "train":
            return loss
    
    def amortized_sampling(self, batch):
        """Amortized sampling of decoder

        Args:
            batch (dict): batch containing observations and conditions
            
        Returns:
            torch.tensor: sample of the decoder
        """
        z = self.encode(batch["X"])["z"]
        return self.sample_decoder(z)

    def random_sampling(self, batch_size):
        """Sample from normal latent codes and decode

        Args:
            batch_size (int): size of the batch to sample 

        Returns:
            torch.tensor: decoded sample
        """
        z = torch.randn(batch_size, self.latent_dim)
        return self.sample_decoder(z)
        
    def sample_decoder(self, z_batch):
        """Sample decoder function given latent codes 

        Args:
            z_batch (torch.tensor): latent codes

        Returns:
            torch.tensor: samples from data distribution
        """
        # Decode a z_output
        decoder_output = self._preprocess_decoder_output(self.decode(z_batch))
        if self.likelihood == "gaussian":
            distr = get_distribution(decoder_output, self.log_sigma, likelihood = self.likelihood)
            return distr.rsample()

        elif self.likelihood == "nb" or self.likelihood == "zinb":
            distr = get_distribution(decoder_output, self.theta, likelihood = self.likelihood)
            return distr.sample()

        else:
            raise NotImplementedError
    
    def on_train_epoch_end(self):
        """Action at the end of training epoch
        """
        if self.anneal_kl:
            self.kl_weight += self.kl_weight_increase
