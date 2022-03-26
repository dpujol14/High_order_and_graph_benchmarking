import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from einops import reduce
import torchmetrics
import torch_geometric

class GraphNeuralNetworkWrapper(pl.LightningModule):
    def __init__(self, hpars, train_idx=None, valid_idx=None, test_idx=None):
        super().__init__()
        self.hpars = hpars
        self.model = self.choose_model(self.hpars.model.name)

        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx

        # Readout function
        self.readout = nn.Linear(self.hpars.experiment.hidden_dim, self.hpars.experiment.n_classes)

        # METRICS
        # Accuracy metrics
        if self.hpars.experiment.classification:
            self.train_metric = torchmetrics.Accuracy()
            self.val_metric = torchmetrics.Accuracy(num_classes=self.hpars.experiment.n_classes)
            self.test_metric = torchmetrics.Accuracy()
        else:
            self.train_metric = torchmetrics.F1Score(self.hpars.experiment.n_classes)
            self.val_metric = torchmetrics.F1Score(self.hpars.experiment.n_classes)
            self.test_metric = torchmetrics.F1Score(self.hpars.experiment.n_classes)

    def choose_model(self, model_name):
        if model_name == 'GINE':
            from models.GINE.model import GINE
            return GINE(self.hpars)
        elif model_name == 'vanilla_GAT':
            from models.vanilla_GAT.models import GAT
            return GAT(self.hpars)
        elif model_name == 'mult_GAT':
            from models.mult_GAT.models import MultGAT
            return MultGAT(self.hpars)
        elif model_name == 'full_neighborhood_mult_GAT':
            from models.full_neighborhood_mult_GAT.models import FullNeighborhoodMultGAT
            return FullNeighborhoodMultGAT(self.hpars)

    def global_pooling_func(self, x, batch, method='mean'):
        if method=='mean':
            return torch_geometric.nn.global_mean_pool(x=x, batch=batch)
        elif method=='max':
            return torch_geometric.nn.global_max_pool(x=x, batch=batch)
        elif method=='add':
            return torch_geometric.nn.global_add_pool(x=x, batch=batch)

    def forward(self, x, edge_attr, edge_idx, batch):
        node_feats = torch.tensor(x, dtype=torch.float)
        edge_feats = torch.tensor(edge_attr, dtype=torch.float)

        # Run the actual GNN model
        x = self.model(node_feats, edge_feats, edge_idx) # x: (N x n_classes)

        # Apply the necessary pooling (depending on the task)
        if self.hpars.experiment.graph_level:
            x = self.global_pooling_func(x, batch)

        x = self.readout(x)
        return x

    def compute_loss(self, y_hat, y):
        if self.hpars.experiment.classification:
            y = torch.squeeze(y)
            return F.cross_entropy(y_hat, y)
        else:
            y_hat = torch.squeeze(y_hat)
            return F.mse_loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        x, edge_attr, edge_idx, batch, y = batch.x, batch.edge_attr, batch.edge_index, batch.batch, batch.y
        y_hat = self.forward(x, edge_attr, edge_idx, batch)

        # Filter out the final predictions and the labels (if node-level task)
        #if self.hpars.experiment.node_level:
        #    y_hat = y_hat[self.train_idx]
        #    y = y[self.train_idx]

        loss = self.compute_loss(y_hat, y)

        self.log('train_loss', loss, on_epoch=True)

        # Compute the metrics
        if self.hpars.experiment.classification:
            y_hat = F.softmax(y_hat, -1)
            self.train_metric(y_hat, torch.squeeze(y, -1))
            self.log('train_accuracy', self.train_metric, on_epoch=True, prog_bar=True, logger=True)
        else:
            self.train_metric(torch.unsqueeze(y_hat,0), torch.squeeze(y,0))
            self.log('train_F1Score', self.train_metric, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_attr, edge_idx, batch, y = batch.x, batch.edge_attr, batch.edge_index, batch.batch, batch.y
        y_hat = self.forward(x, edge_attr, edge_idx, batch)

        # Filter out the final predictions and the labels (if node-level task)
        #if self.hpars.experiment.node_level:
        #    y_hat = y_hat[self.valid_idx]
        #    y = y[self.valid_idx]

        loss = self.compute_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True)

        # Compute the metrics
        if self.hpars.experiment.classification:
            y_hat = F.softmax(y_hat, -1)
            self.val_metric(y_hat, torch.squeeze(y, -1))
            self.log('val_accuracy', self.val_metric, on_epoch=True, prog_bar=True, logger=True)
        else:
            self.val_metric(torch.unsqueeze(y_hat, 0), torch.squeeze(y, 0))
            self.log('val_F1Score', self.val_metric, on_epoch=True, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        x, edge_attr, edge_idx, batch, y = batch.x, batch.edge_attr, batch.edge_index, batch.batch, batch.y
        y_hat = self.forward(x, edge_attr, edge_idx, batch)

        # Filter out the final predictions and the labels (if node-level task)
        #if self.hpars.experiment.node_level:
        #    y_hat = y_hat[self.test_idx]
        #    y = y[self.test_idx]

        loss = self.compute_loss(y_hat, y)
        self.log('test_loss', loss)

        # Compute the metrics
        if self.hpars.experiment.classification:
            y_hat = F.softmax(y_hat, -1)
            self.test_metric(y_hat, torch.squeeze(y, -1))
            self.log('test_accuracy', self.test_metric, on_epoch=True, prog_bar=True, logger=True)
        else:
            self.test_metric(torch.unsqueeze(y_hat, 0), torch.squeeze(y, 0))
            self.log('test_F1Score', self.test_metric, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hpars.optimizer.lr, weight_decay=self.hpars.optimizer.weight_decay)

        return optimizer
