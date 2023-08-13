import sys 
import pytorch_lightning as pl
import seml
import torch
from sacred import SETTINGS, Experiment
from functools import partial

sys.path.insert(0,"../")
from paths import EXPERIMENT_FOLDER

from scCFM.datamodules.time_sc_datamodule import TrajectoryDataModule
from scCFM.models.cfm.cfm_module import CFMLitModule
from scCFM.models.cfm.components.simple_mlp import VelocityNet

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


# Avoid lists in an input configuration to be read-only 
SETTINGS.CONFIG.READ_ONLY_CONFIG = False
SETTINGS['CAPTURE_MODE'] = 'sys'

# Initialize seml experiment
ex = Experiment()
seml.setup_logger(ex)

# Setup the statistics collection post experiment 
@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

# Configure the seml experiment 
@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite))
            
# Training function 
class Solver:
    def __init__(self):
        self.init_all()
    
    def init_all(self):
        self.init_general()
        self.init_datamodule()
        self.init_net()
        self.init_model()
        self.init_checkpoint_callback()
        self.init_early_stopping_callback()
        self.init_logger()
        self.init_trainer()

    @ex.capture(prefix="training")
    def init_general(self, 
                     task_name,
                     seed):
        
        self.task_name = task_name 
        
        # Fix seed for reproducibility
        torch.manual_seed(seed)      
        if seed: 
            pl.seed_everything(seed, workers=True)
            
        # Initialize folder 
        self.current_experiment_dir = EXPERIMENT_FOLDER / self.task_name
        self.current_experiment_dir.mkdir(parents=True, exist_ok=True) 
    
    @ex.capture(prefix="datamodule")
    def init_datamodule(self, 
                        path,
                        x_layer, 
                        time_key,
                        use_pca,
                        n_dimensions, 
                        train_val_test_split,
                        batch_size,
                        num_workers):
        
        # Initialize datamodule
        self.datamodule = TrajectoryDataModule(path=path,
                                               x_layer=x_layer,
                                               time_key=time_key,
                                               use_pca=use_pca,
                                               n_dimensions=n_dimensions,
                                               train_val_test_split=train_val_test_split,
                                               batch_size=batch_size,
                                               num_workers=num_workers)
         
    @ex.capture(prefix="net")
    def init_net(self, 
                 hidden_dims,
                 batch_norm,
                 activation):
        
        # Neural network 
        net_hparams = {"hidden_dims": hidden_dims,
                           "batch_norm": batch_norm,
                           "activation": activation}
        
        self.net = partial(VelocityNet, 
                           **net_hparams)   

    @ex.capture(prefix="model")
    def init_model(self,
                   in_dim, 
                   ot_sampler,
                   sigma,
                   lr,
                   use_real_time, 
                   antithetic_time_sampling):
        
        # Initialize the model 
        self.model = CFMLitModule(
                            in_dim=in_dim,
                            net=self.net,
                            datamodule=self.datamodule,
                            ot_sampler=ot_sampler, 
                            sigma=sigma, 
                            lr=lr, 
                            use_real_time=use_real_time,
                            antithetic_time_sampling=antithetic_time_sampling) 
        
    @ex.capture(prefix="model_checkpoint")
    def init_checkpoint_callback(self, 
                                 filename, 
                                 monitor,
                                 mode,
                                 save_last,
                                 auto_insert_metric_name):
        
        # Initialize callbacks 
        self.model_ckpt_callbacks = ModelCheckpoint(dirpath=self.current_experiment_dir / "checkpoints", 
                                                    filename=filename,
                                                    monitor=monitor,
                                                    mode=mode,
                                                    save_last=save_last,
                                                    auto_insert_metric_name=auto_insert_metric_name)
    
    @ex.capture(prefix="early_stopping")
    def init_early_stopping_callback(self, 
                                     monitor, 
                                     patience, 
                                     mode,
                                     min_delta,
                                     verbose,
                                     strict, 
                                     check_finite,
                                     stopping_threshold,
                                     divergence_threshold,
                                     check_on_train_epoch_end):
        
        # Initialize callbacks 
        self.early_stopping_callbacks = EarlyStopping(monitor=monitor,
                                                      patience=patience, 
                                                      mode=mode,
                                                      min_delta=min_delta,
                                                      verbose=verbose,
                                                      strict=strict,
                                                      check_finite=check_finite,
                                                      stopping_threshold=stopping_threshold,
                                                      divergence_threshold=divergence_threshold,
                                                      check_on_train_epoch_end=check_on_train_epoch_end
                                                      )
        
    @ex.capture(prefix="logger")
    def init_logger(self, 
                    offline, 
                    id, 
                    project, 
                    log_model, 
                    prefix, 
                    group, 
                    tags, 
                    job_type):
        
        # Initialize logger 
        self.logger = WandbLogger(save_dir=self.current_experiment_dir, 
                                  offline=offline,
                                  id=id, 
                                  project=project,
                                  log_model=log_model, 
                                  prefix=prefix,
                                  group=group,
                                  tags=tags,
                                  job_type=job_type) 
        
    @ex.capture(prefix="trainer")
    def init_trainer(self, 
                     max_epochs,
                     accelerator,
                     devices, 
                     log_every_n_steps):    
        
        # Initialize the lightning trainer 
        self.trainer = Trainer(callbacks=[self.model_ckpt_callbacks, self.early_stopping_callbacks], 
                          default_root_dir=self.current_experiment_dir,
                          logger=self.logger, 
                          max_epochs=max_epochs,
                          accelerator=accelerator,
                          devices=devices,
                          log_every_n_steps=log_every_n_steps)
                
    def train(self):
        # Fit the model 
        self.trainer.fit(model=self.model, 
                          train_dataloaders=self.datamodule.train_dataloader(),
                          val_dataloaders=self.datamodule.val_dataloader())
        
        train_metrics = self.trainer.callback_metrics
        return train_metrics

@ex.command(unobserved=True)
def get_experiment(init_all=True):
    print("get_experiment")
    experiment = Solver()
    return experiment

@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = Solver()
    return experiment.train()
