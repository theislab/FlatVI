import sys 
import pytorch_lightning as pl
import seml
import torch
from sacred import SETTINGS, Experiment
from functools import partial

sys.path.insert(0,"../")
from paths import EXPERIMENT_FOLDER

from PerturbSeq_CMV.datamodules.distribution_datamodule import TrajectoryDataModule
from PerturbSeq_CMV.models.cfm_module import CFMLitModule
from PerturbSeq_CMV.models.components.augmentation import AugmentationModule
from PerturbSeq_CMV.models.components.simple_mlp import VelocityNet

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


# Avoid lists in an input configuration to be read-only 
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

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
        self.init_augmentations()
        self.init_net()
        self.init_model()
        self.init_checkpoint_callback()
        self.init_early_stopping_callback()
        self.init_logger()
        self.init_trainer()

    @ex.capture(prefix="training")
    def init_general(self, training):
        self.task_name = training["task_name"] 
        
        # Fix seed for reproducibility
        torch.manual_seed(training["seed"])      
        if training["seed"]: 
            pl.seed_everything(training["seed"], workers=True)
            
        # Initialize folder 
        self.current_experiment_dir = EXPERIMENT_FOLDER / self.task_name
        self.current_experiment_dir.mkdir(parents=True, exist_ok=True) 
    
    @ex.capture(prefix="datamodule")
    def init_datamodule(self, datamodule):
        # Initialize datamodule
        self.datamodule = TrajectoryDataModule(**datamodule)
    
    @ex.capture(prefix="augmentations")
    def init_augmentations(self, augmentations):
        # Initialize augmentations
        self.augmentations = AugmentationModule(**augmentations)
         
    @ex.capture(prefix="net")
    def init_net(self, net):
        # Neural network 
        self.net = partial(VelocityNet, **net)   

    @ex.capture(prefix="model")
    def init_model(self, model):
        # Initialize the model 
        self.model = CFMLitModule(
                            net=self.net,
                            datamodule=self.datamodule,
                            augmentations= self.augmentations, 
                            **model
                            ) 
        
    @ex.capture(prefix="model_checkpoint")
    def init_checkpoint_callback(self, model_checkpoint):
        # Initialize callbacks 
        self.model_ckpt_callbacks = ModelCheckpoint(dirpath=self.current_experiment_dir / "checkpoints", 
                                                **model_checkpoint)
    
    @ex.capture(prefix="early_stopping")
    def init_early_stopping_callback(self, early_stopping):
        # Initialize callbacks 
        self.early_stopping_callbacks = EarlyStopping(**early_stopping)
        
    @ex.capture(prefix="logger")
    def init_logger(self, logger):
        # Initialize logger 
        self.logger = WandbLogger(save_dir=self.current_experiment_dir / "logs", 
                             **logger) 
        
    @ex.capture(prefix="trainer")
    def init_trainer(self, trainer):    
        # Initialize the lightning trainer 
        self.trainer = Trainer(callbacks=[self.model_ckpt_callbacks, self.early_stopping_callbacks], 
                          default_root_dir=self.current_experiment_dir,
                          logger=self.logger, 
                          **trainer)
        
    def train(self):
        # Fit the model 
        self.trainer.fit(model=self.model, 
                          train_dataloaders=self.datamodule.train_dataloader(),
                          val_dataloaders=self.datamodule.val_dataloader())
        train_metrics = self.trainer.callback_metrics
        
        # Test model 
        ckpt_path = self.trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            ckpt_path = None
        self.trainer.test(model=self.model, 
                          dataloaders=self.datamodule.test_dataloader(),
                          ckpt_path=ckpt_path)
        test_metrics = self.trainer.callback_metrics

        # merge train and test metrics
        metric_dict = {**train_metrics, **test_metrics}
        
        return metric_dict

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
