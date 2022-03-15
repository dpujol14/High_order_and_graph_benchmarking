import sys
import os

if os.path.isdir(os.path.abspath("../..")):
    sys.path.append(os.path.abspath("../.."))
else:
    sys.path.append(os.path.abspath("../.."))

from load_data import load_dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint,EarlyStopping, DeviceStatsMonitor
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
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
    dataset_name = "ogbg-molpcba"
    data_path = os.path.join(os.path.abspath(hpars.experiment.path_to_data), dataset_name)
    os.makedirs(data_path, exist_ok=True)
    train_loader, valid_loader, test_loader = load_dataset(data_path=data_path, dataset_name= dataset_name, batch_size=8)

    # Load the model
    model = GraphNeuralNetworkWrapper(hpars)

    callbacks = [LearningRateMonitor(),
                 DeviceStatsMonitor(),
                 ModelCheckpoint(monitor=hpars.early_stop_metric,
                                 filename='{epoch}-{train_loss_epoch:.2f}--{val_loss_epoch:.2f}',
                                 save_top_k=1,
                                 save_last=True)]

    BACKEND = 'dp'
    my_logger = TensorBoardLogger('tb_logs', name=hpars.experiment.exp_name + '_' + hpars.model.name)
    # alt_logger = MLFlowLoggerCheckpointer("neural_sorting_net", tracking_uri=MLFLOW_URI),
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
