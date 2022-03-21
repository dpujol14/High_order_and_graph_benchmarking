import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from einops import reduce
import torchmetrics

class GraphNeuralNetworkWrapper(pl.LightningModule):
    def __init__(self, hpars, train_idx=None, valid_idx=None, test_idx=None):
        super().__init__()
        self.hpars = hpars
        self.model = self.choose_model(self.hpars.model.name)

        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx

        # METRICS
        # Accuracy metrics
        self.train_acc_metric = torchmetrics.Accuracy()
        self.val_acc_metric = torchmetrics.Accuracy()

    def choose_model(self, model_name):
        if model_name == 'vanilla_GAT':
            from models.vanilla_GAT.models import GAT
            return GAT(self.hpars)
        elif model_name == 'mult_GAT':
            from models.mult_GAT.models import MultGAT
            return MultGAT(self.hpars)
        elif model_name == 'full_neighborhood_mult_GAT':
            from models.full_neighborhood_mult_GAT.models import FullNeighborhoodMultGAT
            return FullNeighborhoodMultGAT(self.hpars)

    def pooling_func(self, x, method='mean'):
        return reduce(x, 'n d -> d', method)

    def forward(self, node_feats, edge_feats, edge_indices, adj):
        node_feats = torch.tensor(node_feats, dtype=torch.float)
        edge_feats = torch.tensor(edge_feats, dtype=torch.float)

        # Run the actual GNN model
        x = self.model(node_feats, edge_feats, edge_indices, adj) # x: (N x n_classes)

        # Apply the necessary pooling (depending on the task)
        if self.hpars.experiment.graph_level:
            x = self.pooling_func(x)

        return x

    def compute_loss(self, y_hat, y):
        if self.hpars.experiment.classification:
            y = torch.squeeze(y)
            return F.cross_entropy(y_hat, y)
        else:
            y_hat = torch.squeeze(y_hat)
            return F.mse_loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        graph, y = batch

        # Obtain the data from the graph
        adj = graph.adj().to_dense()
        node_feats = graph.ndata['feat']    # n_nodes x f
        edge_feats = graph.edata['feat']    # n_edges x f'
        edge_indices = graph.edges()

        y_hat = self.forward(node_feats, edge_feats, edge_indices, adj)

        # Filter out the final predictions and the labels (if node-level task)
        if self.hpars.experiment.node_level:
            y_hat = y_hat[self.train_idx]
            y = y[self.train_idx]

        loss = self.compute_loss(y_hat, y)

        self.log('train_loss', loss, on_epoch=True)

        # Compute the accuracy
        y_hat = F.softmax(y_hat)
        acc = self.metric_acc(torch.unsqueeze(y_hat,0), torch.squeeze(y,0))
        self.log('train_accuracy', acc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        graph, y = batch

        # Obtain the data from the graph
        adj = graph.adj().to_dense()
        node_feats = graph.ndata['feat']
        edge_feats = graph.edata['feat']
        edge_indices = graph.edges()

        y_hat = self.forward(node_feats, edge_feats, edge_indices, adj)

        # Filter out the final predictions and the labels (if node-level task)
        if self.hpars.experiment.node_level:
            y_hat = y_hat[self.valid_idx]
            y = y[self.valid_idx]

        loss = self.compute_loss(y_hat, y)

        self.log('val_loss', loss, on_step=True)

        y_hat = F.softmax(y_hat)
        acc = self.val_acc_metric(torch.unsqueeze(y_hat,0), torch.squeeze(y,0))
        self.log('val_accuracy', acc, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        graph, y = batch

        # Obtain the data from the graph
        adj = graph.adj().to_dense()
        node_feats = graph.ndata['feat']
        edge_feats = graph.edata['feat']
        edge_indices = graph.edges()

        y_hat = self.forward(node_feats, edge_feats, edge_indices, adj)

        # Filter out the final predictions and the labels (if node-level task)
        if self.hpars.experiment.node_level:
            y_hat = y_hat[self.test_idx]
            y = y[self.test_idx]

        loss = self.compute_loss(y_hat, y)

        self.log('test_loss', loss)

        # Compute the accuracy
        y_hat = F.softmax(y_hat)
        acc = self.metric_acc(torch.unsqueeze(y_hat,0), torch.squeeze(y,0))
        self.log('test_accuracy', acc, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hpars.optimizer.lr, weight_decay=self.hpars.optimizer.weight_decay)

        return optimizer
