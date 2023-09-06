import torch
from scCFM.models.utils import jacobian_decoder_jvp_parallel
from scCFM.models.manifold.geometry_metrics import compute_all_metrics
from scCFM.models.base.vae import VAE
    
class GeometricNBVAE(VAE):
    def __init__(self, 
                 l2, 
                 fl_weight, 
                 interpolate_z,
                 eta_interp,
                 compute_metrics_every,
                 start_jac_after,
                 vae_kwargs, 
                 use_c, 
                 detach_theta):
        """Geometric Variational Autoencdoer 

        Args:
            l2 (bool): whether to use l2 loss on the metric
            fl_weight (float): the weight for the flattening loss
            interpolate_z (bool): whether to interpolate the values of the batch
            eta_interp (float): the lower/upper bound of the uniform distribution 
            compute_metrics_every (int): how often to compute the metrics
            start_jac_after (int): after how many epochs optimize
            vae_kwargs (dict): dictionary containing basic variational autoencoder parameters 
            use_c (bool): use trace of the metric tensor on the diagonal 
            detach_theta (bool): whether to detach from the gradient of the flattening loss
        """
        super(GeometricNBVAE, self).__init__(**vae_kwargs)
 
        assert vae_kwargs["likelihood"] == "nb" or vae_kwargs["likelihood"] == "zinb"
        self.l2 = l2
        self.fl_weight = fl_weight
        self.interpolate_z = interpolate_z
        self.eta_interp = eta_interp
        self.compute_metrics_every = compute_metrics_every
        self.start_jac_after = start_jac_after
        self.use_c = use_c
        self.detach_theta = detach_theta
        
        self.n_epochs_so_far = 0  # keep record of epochs performed 
        
    def step(self, batch, prefix):
        """Compute losses and log the results

        Args:
            batch (torch.tensor): _description_
            prefix (str): train or valid
        """
        # Read batch
        x = batch["X"]

        # Compute encoder-decoder input and output
        decoder_outputs, z, mu, logvar = self.forward(batch)
        decoder_outputs = self._preprocess_decoder_output(decoder_outputs)
        
        # Compute a reconstruction loss 
        recon_loss = self.reconstruction_loss(x, decoder_outputs)
        
        # Compute kl divergence
        kl_div = self.kl_divergence(mu, logvar)
        kl_weight = min([self.kl_weight, 1])
    
        # Compute Jacobian regualrization 
        if self.interpolate_z and prefix not in ["val", "train"]:
            z = self.augment_data(z)
        fisher_rao_metric = self.metric(z, decoder_outputs)
        fl_loss = self.flattening_loss(z, fisher_rao_metric)
        
        # Loss function
        if self.fl_weight == 0 or self.n_epochs_so_far < self.start_jac_after:
            loss = torch.mean(recon_loss + kl_weight * kl_div)
        else:
            loss = torch.mean(recon_loss + kl_weight * kl_div + self.fl_weight * fl_loss)
            
        dict_losses = {f"{prefix}/loss": loss,
                       f"{prefix}/kl": kl_div.mean(),
                       f"{prefix}/lik": recon_loss.mean(), 
                       f"{prefix}/fl_loss": fl_loss.mean()}
        
        self.log_dict(dict_losses, prog_bar=True)
        
        if prefix in ["val", "test"] and self.n_epochs_so_far % self.compute_metrics_every == 0:
            dict_metrics = compute_all_metrics(fisher_rao_metric, z, decoder_outputs, self)
            self.log_dict(dict_metrics, prog_bar=True)

        if prefix == "train":
            return loss            
    
    def flattening_loss(self, z, metric_tensor):
        """Loss for making the manifold flat 

        Args:
            z (torch.tensor): latent point where curvature is evaluated
            metric_tensor (torch.tensor): metric tensor representing local curvature 

        Returns:
            torch.tensor: tensor with the flattening loss
        """
        # Get constant scaling term 
        if self.use_c:
            n_z = z.shape[-1]
            c = torch.mean(1/n_z * metric_tensor.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1))
        else:
            c = 1 
        
        # Compute identity 
        Id =  torch.eye(z.shape[1])
        Id = Id.unsqueeze(0).expand(z.shape[0], z.shape[1], z.shape[1]).to(self.device)  # B x d x d
        
        # Compute regularization loss 
        if self.l2:
            loss = torch.sum((metric_tensor - c * Id)**2, dim=(1, 2))
        else:
            loss = torch.sum(torch.abs(metric_tensor - c * Id), dim=(1, 2))
        return loss 
    
    def augment_data(self, z):
        """Augment data 

        Args:
            z (torch.tensor): latent point tensor to optimize

        Returns:
            torch.tensor: shuffled values of z
        """
        # Linear interpolation between pairs of z
        bs, _ = z.size()
        z_permuted = z[torch.randperm(bs)]
        alpha_samples = (torch.rand(bs, 1) * (1 + 2 * self.eta_interp) - self.eta_interp).to(self.device)
        z_augmented = alpha_samples * z + (1 - alpha_samples) * z_permuted
        return z_augmented
    
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
        theta = torch.exp(self.theta).detach() if self.detach_theta else torch.exp(self.theta)
        
        if self.likelihood == "nb":
            nb_fisher = theta * 1 / (decoder_outputs["mu"] * (theta + decoder_outputs["mu"]))  # B x D
        else:
            raise NotImplementedError
        
        if self.model_library_size:
            jac = jacobian_decoder_jvp_parallel(self.decode, z, v=None, create_graph=True)[:, :-1, :]
        else:
            jac = jacobian_decoder_jvp_parallel(self.decode, z, v=None, create_graph=True)

        return torch.einsum("bij,bik->bjk", jac, jac * nb_fisher.unsqueeze(-1))
    
    def on_train_epoch_end(self):
        """Increase epoch count and weights
        """
        # Increase epochs
        self.n_epochs_so_far += 1
        # Increase kl weight
        if self.anneal_kl:
            self.kl_weight += self.kl_weight_increase

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
            