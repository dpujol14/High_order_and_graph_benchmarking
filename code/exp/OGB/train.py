import sys
import os

if os.path.isdir(os.path.abspath("../..")):
    sys.path.append(os.path.abspath("../.."))
else:
    sys.path.append(os.path.abspath("../.."))

from load_data import load_graph_level_dataset#, load_node_level_dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig
from codecarbon import EmissionsTracker
from models.model_wrapper import GraphNeuralNetworkWrapper


@hydra.main(config_path='../configurations', config_name='OGB')
def main(hpars:DictConfig):
    """
    A sample experimental setup for the LongRangeArena
    """
    pl.seed_everything(hpars.seed)

    # Load the dataset
    print("Loading the data...")
    data_path = os.path.join(os.path.abspath(hpars.experiment.path_to_data), hpars.experiment.exp_name)
    os.makedirs(data_path, exist_ok=True)

    train_idx, valid_idx, test_idx = None, None, None
    if hpars.experiment.graph_level:
        train_loader, valid_loader, test_loader = load_graph_level_dataset(data_path=data_path, dataset_name= hpars.experiment.exp_name, batch_size=hpars.batch_size)

    elif hpars.experiment.node_level:
        data_loader, train_idx, valid_idx, test_idx = load_node_level_dataset(data_path=data_path, dataset_name= hpars.experiment.exp_name, batch_size=hpars.batch_size)
        train_loader = data_loader
        valid_loader = data_loader
        test_loader = data_loader

    # Load the model
    model = GraphNeuralNetworkWrapper(hpars, train_idx, valid_idx, test_idx)

    callbacks = [LearningRateMonitor(),
                 DeviceStatsMonitor(),
                 ModelCheckpoint(monitor=hpars.early_stop_metric,
                                 filename='{epoch}-{train_loss_epoch:.2f}--{val_loss_epoch:.2f}',
                                 save_top_k=1,
                                 save_last=True)]

    BACKEND = 'dp'
    my_logger = TensorBoardLogger('tb_logs', name=hpars.experiment.exp_name + '_' + hpars.model.name)

    trainer = pl.Trainer(logger=my_logger,
                         max_epochs=hpars.experiment.epochs,
                         gpus=None if hpars.gpus == -1 else hpars.gpus,
                         accelerator=BACKEND if hpars.gpus else None,
                         callbacks=callbacks,
                         weights_summary="full")  # weights summary is highly useful for quick debugging

    print("Starting the training...")
    tracker = EmissionsTracker()
    tracker.start()
    trainer.fit(model, train_loader, valid_loader)
    tracker.stop()

if __name__=="__main__":
    main()
