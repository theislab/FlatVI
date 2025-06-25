import sys 
import pytorch_lightning as pl
import torch
import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0,"../")
from paths import EXPERIMENT_FOLDER

from scCFM.datamodules.time_sc_datamodule import TrajectoryDataModule
from scCFM.models.cfm.cfm_module import CFMLitModule
from scCFM.models.cfm.components.mlp import MLP

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from scCFM.train_hydra.utils.exceptions import *
 
# Filter out torch warnings 
warnings.filterwarnings(
    "ignore",
    "There is a wandb run already in progress",
    module="pytorch_lightning.loggers.wandb",
)

@hydra.main(config_path="../../config_hydra", config_name="train", version_base=None)
@print_exceptions
def main(config: DictConfig):
    # Resolve interpolations to work around a bug:
    # https://github.com/omry/omegaconf/issues/862
    OmegaConf.resolve(config)
    task_name = config.train.task_name 
         
    # Initialize folder 
    current_experiment_dir = EXPERIMENT_FOLDER / task_name
    current_experiment_dir.mkdir(parents=True, exist_ok=True)     
        
    # Initialize datamodule
    datamodule = TrajectoryDataModule(path=config.datamodule.path,
                                        x_layer=config.datamodule.x_layer,
                                        time_key=config.datamodule.time_key,
                                        use_pca=config.datamodule.use_pca,
                                        n_dimensions=config.datamodule.n_dimensions,
                                        train_val_test_split=config.datamodule.train_val_test_split,
                                        batch_size=config.datamodule.batch_size,
                                        num_workers=config.datamodule.num_workers, 
                                        model_library_size=config.datamodule.model_library_size)
        
    # Neural network 
    net_hparams = {"dim": datamodule.dim,
                    "w": config.net.w,
                    "time_varying": config.net.time_varying}
    
    net = MLP(**net_hparams) 
        
    # Initialize the model 
    model = CFMLitModule(
                        net=net,
                        datamodule=datamodule,
                        ot_sampler=config.model.ot_sampler, 
                        sigma=config.model.sigma, 
                        lr=config.model.lr, 
                        use_real_time=config.model.use_real_time,
                        antithetic_time_sampling=config.model.antithetic_time_sampling, 
                        leaveout_timepoint=config.model.leaveout_timepoint) 

        
    # Initialize callbacks 
    model_ckpt_callbacks = ModelCheckpoint(dirpath=current_experiment_dir / "checkpoints", 
                                                filename=config.checkpoint.filename,
                                                monitor=config.checkpoint.monitor,
                                                mode=config.checkpoint.mode,
                                                save_last=config.checkpoint.save_last,
                                                auto_insert_metric_name=config.checkpoint.auto_insert_metric_name)
 

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
                        max_steps=config.trainer.max_steps,
                        accelerator=config.trainer.accelerator,
                        devices=config.trainer.devices,
                        log_every_n_steps=config.trainer.log_every_n_steps)
            
    # Fit the model 
    trainer.fit(model=model, 
                        train_dataloaders=datamodule.train_dataloader(),
                        val_dataloaders=datamodule.val_dataloader())

if __name__=="__main__":
    main()
