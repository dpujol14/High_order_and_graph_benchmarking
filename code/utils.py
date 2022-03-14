import os

from pytorch_lightning.loggers import MLFlowLogger

MLFLOW_URI=os.path.join(os.path.expanduser("~"),"mlflow_local")
DATA_DIR=os.path.join(os.path.expanduser("~"),".lions_data")
os.makedirs(MLFLOW_URI,exist_ok=True)
os.makedirs(DATA_DIR,exist_ok=True)


class MLFlowLoggerCheckpointer(MLFlowLogger):
    # from https://stackoverflow.com/questions/59149725/how-can-i-save-model-weights-to-mlflow-tracking-sever-using-pytorch-lightning
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def after_save_checkpoint(self, model_checkpoint) -> None:
        """
        Called after model checkpoint callback saves a new checkpoint. Might have to be refined to not log excessively
        but given as a starting point.
        """
        print("Checkpoint saved")
        self.experiment.log_artifact(
            self.run_id, model_checkpoint.best_model_path
        )