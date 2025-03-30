import sys 
import pytorch_lightning as pl
import torch
import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0,"../")
from paths import EXPERIMENT_FOLDER

from scCFM.datamodules.sc_datamodule import scDataModule
from scCFM.models.base.vae import VAE, AE
from scCFM.models.base.geometric_vae import GeometricNBAE,GeometricNBVAE

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# Filter out torch warnings 
warnings.filterwarnings(
    "ignore",
    "There is a wandb run already in progress",
    module="pytorch_lightning.loggers.wandb",
)

@hydra.main(config_path="../../config_hydra", config_name="train", version_base=None)
# Training function 
def main(config: DictConfig):
    OmegaConf.resolve(config)
    task_name = config.train.task_name 
        
    # Initialize folder 
    current_experiment_dir = EXPERIMENT_FOLDER / task_name
    current_experiment_dir.mkdir(parents=True, exist_ok=True) 

    
    # Initialize datamodule
    datamodule = scDataModule(path=config.datamodule.path,
                                x_layer=config.datamodule.x_layer,
                                cond_keys=config.datamodule.cond_keys,
                                use_pca=config.datamodule.use_pca,
                                n_dimensions=config.datamodule.n_dimensions,
                                train_val_test_split=config.datamodule.train_val_test_split,
                                batch_size=config.datamodule.batch_size,
                                num_workers=config.datamodule.num_workers) 

    if config.model.model_type == "vae":
        # Initialize the model 
        model = VAE(in_dim=datamodule.in_dim,
                    hidden_dims=config.model.hidden_dims,
                    batch_norm=config.model.batch_norm,
                    dropout=config.model.dropout,
                    dropout_p=config.model.dropout_p,
                    n_epochs_anneal_kl=config.model.n_epochs_anneal_kl,
                    kl_warmup_fraction=config.model.kl_warmup_fraction,
                    kl_weight=config.model.kl_weight, 
                    likelihood=config.model.likelihood, 
                    learning_rate=config.model.learning_rate,
                    model_library_size=config.model.model_library_size) 
        
    elif config.model.model_type == "ae":
        model = AE(in_dim=datamodule.in_dim,
                        hidden_dims=config.model.hidden_dims,
                        batch_norm=config.model.batch_norm, 
                        dropout=config.model.dropout,
                        dropout_p=config.model.dropout_p,
                        likelihood=config.model.likelihood, 
                        learning_rate=config.model.learning_rate,
                        model_library_size=config.model.model_library_size)
    
    elif config.model.model_type in ["geometric_ae", "geometric_vae"]:
        if config.model.model_type == "geometric_ae":
            vae_kwargs = dict(in_dim=datamodule.in_dim,
                            hidden_dims=config.model.hidden_dims,
                            batch_norm=config.model.batch_norm,
                            dropout=config.model.dropout,
                            dropout_p=config.model.dropout_p,
                            kl_weight=config.model.kl_weight, 
                            likelihood=config.model.likelihood, 
                            learning_rate=config.model.learning_rate, 
                            model_library_size=config.model.model_library_size)

        elif config.model.model_type == "geometric_vae":   
            vae_kwargs = dict(in_dim=datamodule.in_dim,
                            hidden_dims=config.model.hidden_dims,
                            batch_norm=config.model.batch_norm,
                            dropout=config.model.dropout,
                            dropout_p=config.model.dropout_p,
                            n_epochs_anneal_kl=config.model.n_epochs_anneal_kl,
                            kl_warmup_fraction=config.model.kl_warmup_fraction,
                            kl_weight=config.model.kl_weight, 
                            likelihood=config.model.likelihood, 
                            learning_rate=config.model.learning_rate, 
                            model_library_size=config.model.model_library_size)
    
        
    
    if config.model.model_type == "geometric_ae":
        model = GeometricNBAE(l2=config.geometric_vae.l2,
                                interpolate_z=config.geometric_vae.interpolate_z,
                                eta_interp=config.geometric_vae.eta_interp,
                                start_jac_after=config.geometric_vae.start_jac_after,
                                use_c=config.geometric_vae.use_c,
                                compute_metrics_every=config.geometric_vae.compute_metrics_every,
                                vae_kwargs=vae_kwargs, 
                                detach_theta=config.geometric_vae.detach_theta, 
                                fl_weight=config.geometric_vae.fl_weight,
                                trainable_c=config.geometric_vae.trainable_c,
                                anneal_fl_weight=config.geometric_vae.anneal_fl_weight, 
                                max_fl_weight=config.geometric_vae.max_fl_weight,
                                n_epochs_anneal_fl=config.geometric_vae.n_epochs_anneal_fl, 
                                fl_anneal_fraction=config.geometric_vae.fl_anneal_fraction)
    else:   
        model = GeometricNBVAE(l2=config.geometric_vae.l2,
                                interpolate_z=config.geometric_vae.interpolate_z,
                                eta_interp=config.geometric_vae.eta_interp,
                                start_jac_after=config.geometric_vae.start_jac_after,
                                use_c=config.geometric_vae.use_c,
                                compute_metrics_every=config.geometric_vae.compute_metrics_every,
                                vae_kwargs=vae_kwargs, 
                                detach_theta=config.geometric_vae.detach_theta,
                                fl_weight=config.geometric_vae.fl_weight,
                                trainable_c=config.geometric_vae.trainable_c,
                                anneal_fl_weight=config.geometric_vae.anneal_fl_weight, 
                                max_fl_weight=config.geometric_vae.max_fl_weight,
                                n_epochs_anneal_fl=config.geometric_vae.n_epochs_anneal_fl, 
                                fl_anneal_fraction=config.geometric_vae.fl_anneal_fraction)
    
    
    # Initialize callbacks 
    model_ckpt_callbacks = ModelCheckpoint(dirpath=current_experiment_dir / "checkpoints", 
                                                filename=config.checkpoint.filename,
                                                monitor=config.checkpoint.monitor,
                                                mode=config.checkpoint.mode,
                                                save_last=config.checkpoint.save_last,
                                                auto_insert_metric_name=config.checkpoint.auto_insert_metric_name)

    # Initialize callbacks 
    if config.early_stopping.perform_early_stopping:
        early_stopping_callbacks = EarlyStopping(monitor=config.early_stopping.monitor,
                                                    patience=config.early_stopping.patience, 
                                                    mode=config.early_stopping.mode,
                                                    min_delta=config.early_stopping.min_delta,
                                                    verbose=config.early_stopping.verbose,
                                                    strict=config.early_stopping.strict,
                                                    check_finite=config.early_stopping.check_finite,
                                                    stopping_threshold=config.early_stopping.stopping_threshold,
                                                    divergence_threshold=config.early_stopping.divergence_threshold,
                                                    check_on_train_epoch_end=config.early_stopping.check_on_train_epoch_end
                                                    )
    else:
        early_stopping_callbacks = None
    
    # Initialize logger 
    logger = WandbLogger(save_dir=current_experiment_dir, 
                            offline=config.logger.offline,
                            id=config.logger.id, 
                            project=config.logger.project,
                            log_model=config.logger.log_model, 
                            prefix=config.logger.prefix,
                            group=config.logger.group,
                            tags=config.logger.tags,
                            job_type=config.logger.job_type) 

    if early_stopping_callbacks != None:
        callbacks = [model_ckpt_callbacks, early_stopping_callbacks]
    else:
        callbacks = model_ckpt_callbacks
    # Initialize the lightning trainer 
    trainer = Trainer(callbacks=callbacks, 
                        default_root_dir=current_experiment_dir,
                        logger=logger, 
                        max_epochs=config.trainer.max_epochs,
                        accelerator=config.trainer.accelerator,
                        devices=config.trainer.devices,
                        log_every_n_steps=config.trainer.log_every_n_steps)
            
    # Fit the model 
    trainer.fit(model=model, 
                train_dataloaders=datamodule.train_dataloader(),
                val_dataloaders=datamodule.val_dataloader())
    

if __name__=="__main__":
    main()