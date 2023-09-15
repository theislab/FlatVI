import graphtools
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import csr_matrix
import numpy as np
import torch
import torch.nn as nn
from scCFM.models.base.mlp import MLP
from scCFM.models.base.vae import AE

class DiffusionDistance:
    """
    class DiffusionDistance        
        X (np.array) data 
        t_max (int), 2^t_max is the max scale of the Diffusion kernel
        knn (int) = 5 number of neighbors for the KNN in the alpha decay kernel construction, same default as in PHATE
        Anisotropy (int): the alpha in Coifman Lafon 2006, 1: double normalization 0: usual random walk
        log (bool) log(P) or not
        normalize (bool) min-max normalization of the distance matrix
        phate (bool) is PHATE op if true (should be the same as graphtool)
        
    """
   
    def __init__(self, t_max=5, knn=5, anisotropy=1, log=False, normalize=False, symmetrize=False) -> None:
        self.t_max = t_max
        self.knn = knn
        self.aniso = anisotropy
        self.log = log
        self.normalize = normalize
        self.symmetrize = symmetrize
        self.K = None
        self.P = None
        self.pi = None
        self.G = None    
            
    def compute_stationnary_distrib(self): 
        pi = np.sum(self.K, axis = 1)
        self.pi = (pi/np.sum(pi)).reshape(-1,1)
        return self.pi
        
    def compute_custom_diffusion_distance(self): 
        P = self.P
        P_d = P if not self.log else csr_matrix((np.log(P.data),P.indices,P.indptr), shape=P.shape)
        G = pairwise_distances(P_d,P_d,metric='l1',n_jobs=-1)
                
        for t in range(1,self.t_max): 
            P = P @ P 
            if self.log==True:
                dist = pairwise_distances(P,P,metric='l1',n_jobs=-1)
                np.fill_diagonal(dist,1)
                dist = (-1)*np.log(dist)
            else:
                dist = pairwise_distances(P_d,P_d,metric='l1',n_jobs=-1)
            G = G + 2**(-t/2.0) * dist
        
        if self.log==True:
            dist = pairwise_distances(self.pi,self.pi,metric='l1',n_jobs=-1)
            np.fill_diagonal(dist,1)
            dist = (-1)*np.log(dist)
        else:
            dist = pairwise_distances(self.pi,self.pi,metric='l1',n_jobs=-1)     
        G = G + 2**(-(self.t_max+1)/2.0) * dist
        self.G = G if not self.normalize else (G - np.min(G))/(np.max(G)-np.min(G))
        return self.G

    def fit(self, X):
        graph = graphtools.Graph(X, knn=self.knn,anisotropy=self.aniso)
        self.K = graph.K.toarray()
        self.P = graph.diff_op.toarray() 
        self.compute_stationnary_distrib()
        self.compute_custom_diffusion_distance()       
        return self.G if not self.symmetrize else (self.G + np.transpose(self.G))/0.5

class GeodesicAE(AE):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        batch_norm,
        dropout,
        dropout_p,
        activation=torch.nn.ELU,
        likelihood="nb", 
        learning_rate=1e-4
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
            model_library_size, 
            library_size_regression, 
            data_library_size
        )

        self.latent_layer = torch.nn.Linear(hidden_dims[-2], self.latent_dim)
        # Diffusion distance
        self.dist = DiffusionDistance()
        self.criterion = nn.MSELoss()

        self.decoder_layers = MLP(
            hidden_dims=[*hidden_dims[::-1]],
            batch_norm=batch_norm,
            dropout=dropout,
            dropout_p=dropout_p,
            activation=activation,
        )
    
    def decode(self, z):
        """Decoder function

        Args:
            z (torch.tensor): the input latent space

        Returns:
            torch.tensor: output of the decoder
        """
        # Decode z layers
        x = self.decoder_layers(z)
        return x
    
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
        return self.decode(z), z
    
    def step(self, batch, prefix):
        """Step of the trained model 

        Args:
            batch (dict): batch containing observations and conditions
            prefix (str): prefix of the type of pass (training or validation)

        Returns:
            torch.tensor: loss
        """
        x = batch["X"]
        decoder_output, z = self.forward(batch)
        
        # Reconstruction loss 
        recon_loss = self.criterion(torch.log1p(x), decoder_output)
        
        # Measure diffusion distance
        dist_geo = self.dist.fit(x.cpu().numpy())
        dist_geo = torch.from_numpy(dist_geo).float().to(x.device)
        dist_emb = torch.cdist(z, z)**2
        loss_dist = self.criterion(dist_emb,dist_geo)
        
        # Total loss
        loss = recon_loss + loss_dist 

        loss = torch.mean(loss)
        
        dict_losses = {f"{prefix}/recon_loss": recon_loss,
                        f"{prefix}/kl": loss_dist.mean()}
        self.log_dict(dict_losses, prog_bar=True)
        if prefix == "train":
            return loss
    