import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class GraphNeuralNetworkWrapper(pl.LightningModule):
    def __init__(self, hpars):
        super().__init__()
        self.hpars = hpars
        self.model = self.choose_model(self.hpars.model.name)

    def choose_model(self, model_name):
        #if model_name == 'vanilla_GCN':
        #    from code.model.GCN.model import GCN
        #    return GCN(self.hpars)
        if model_name == 'vanilla_GAT':
            from models.GAT.models import GAT
            return GAT(self.hpars)

    def forward(self, node_feats, adj):
        # Run the actual GNN model
        x = self.model(node_feats, adj)

        # Apply the necessary pooling (depending on the task)

        return x

    def compute_loss(self, y_hat, y):
        if self.hpars.experiment.classification:
            return F.cross_entropy(y_hat, y)
        else:
            y_hat = torch.squeeze(y_hat)
            return F.mse_loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        graph, y = batch

        # Obtain the data from the graph
        adj = graph.adj
        node_feats = graph.ndata['feat']
        edge_feats = graph.edata['feat']

        y_hat = self.forward(graph)

        loss = self.compute_loss(y_hat, y)

        self.log('train_loss', loss, on_epoch=True)

        # Compute the accuracy
        acc = self.metric_acc(y_hat, y)
        self.log('train_accuracy', acc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        graph, y = batch

        # Obtain the data from the graph
        adj = graph.adj
        node_feats = graph.ndata['feat']
        edge_feats = graph.edata['feat']

        y_hat = self.forward(node_feats, adj)

        loss = self.compute_loss(y_hat, y)

        self.log('val_loss', loss, on_step=True)

        acc = self.metric_acc(y_hat, y)
        self.log('val_accuracy', acc, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        graph, y = batch

        # Obtain the data from the graph
        adj = graph.adj
        node_feats = graph.ndata['feat']
        edge_feats = graph.edata['feat']

        y_hat = self.forward(graph)

        loss = self.compute_loss(y_hat, y)

        self.log('test_loss', loss)

        # Compute the accuracy
        acc = self.metric_acc(y_hat, y)
        self.log('test_accuracy', acc, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hpars.optimizer.lr, weight_decay=self.hpars.optimizer.weight_decay)

        return optimizer
