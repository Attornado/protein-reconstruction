# This implementation is based on the one from the repository:
# https://github.com/diningphil/gnn-comparison, all rights reserved to authors and contributors.
# Copyright (C)  2020  University of Pisa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import os
from typing import Callable, Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.aggr import SortAggregation
from torch_geometric.utils import add_self_loops, degree
from torchmetrics.functional import accuracy, precision, recall, f1_score

from models.classification.classifiers import GraphClassifier
from models.layers import SerializableModule
from training.training_tools import FIGURE_SIZE_DEFAULT, MetricsHistoryTracer, EarlyStopping, EARLY_STOP_PATIENCE


class DGCNN(GraphClassifier):
    """
    Uses fixed architecture
    """

    def __init__(self, dim_features, dim_target, config):
        super(DGCNN, self).__init__(dim_features, dim_target, config)

        self.ks = {
            'NCI1': {'0.6': 30, '0.9': 46},
            'PROTEINS_full': {'0.6': 32, '0.9': 81},
            'DD': {'0.6': 291, '0.9': 503},
            'ENZYMES': {'0.6': 36, '0.9': 48},
            'IMDB-BINARY': {'0.6': 18, '0.9': 31},
            'IMDB-MULTI': {'0.6': 11, '0.9': 22},
            'REDDIT-BINARY': {'0.6': 370, '0.9': 1002},
            'REDDIT-MULTI-5K': {'0.6': 469, '0.9': 1081},
            'COLLAB': {'0.6': 61, '0.9': 130},
            'PSCDB': {'0.6': 327, '0.9': 600}  # was 327, 429
        }

        self.k = self.ks[config['dataset']][str(config['k'])]
        self.embedding_dim = config['embedding_dim']
        self.num_layers = config['num_layers']

        self.convs = []
        for layer in range(self.num_layers):
            input_dim = dim_features if layer == 0 else self.embedding_dim
            self.convs.append(DGCNNConv(input_dim, self.embedding_dim))
        self.total_latent_dim = self.num_layers * self.embedding_dim

        # Add last embedding
        self.convs.append(DGCNNConv(self.embedding_dim, 1))
        self.total_latent_dim += 1

        self.convs = nn.ModuleList(self.convs)

        # should we leave this fixed?
        self.conv1d_params1 = nn.Conv1d(1, 16, self.total_latent_dim, self.total_latent_dim)
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.global_sort_pool = SortAggregation(k=self.k)
        self.conv1d_params2 = nn.Conv1d(16, 32, 5, 1)

        dense_dim = int((self.k - 2) / 2 + 1)
        self.input_dense_dim = (dense_dim - 5 + 1) * 32

        self.hidden_dense_dim = config['dense_dim']
        self.dense_layer = nn.Sequential(nn.Linear(self.input_dense_dim, self.hidden_dense_dim),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(self.hidden_dense_dim, dim_target))

    def forward(self, x, edge_index, batch):
        # Implement Equation 4.2 of the paper i.e. concat all layers' graph representations and apply linear model
        # note: this can be decomposed in one smaller linear model per layer
        # x, edge_index, batch = data.x, data.edge_index, data.batch

        hidden_repres = []

        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            hidden_repres.append(x)

        # apply sortpool
        x_to_sortpool = torch.cat(hidden_repres, dim=1)
        x_1d = self.global_sort_pool(x_to_sortpool, batch)  # in the code the authors sort the last channel only

        # apply 1D convolutional layers
        x_1d = torch.unsqueeze(x_1d, dim=1)
        conv1d_res = F.relu(self.conv1d_params1(x_1d))
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = F.relu(self.conv1d_params2(conv1d_res))
        conv1d_res = conv1d_res.reshape(conv1d_res.shape[0], -1)

        # apply dense layer
        out_dense = self.dense_layer(conv1d_res)
        return out_dense


# noinspection PyAbstractClass
class DGCNNConv(MessagePassing):
    """
    Extended from tutorial on GCNs of Pytorch Geometrics
    """

    def __init__(self, in_channels, out_channels):
        super(DGCNNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    # noinspection PyMethodOverriding
    def message(self, x_j: torch.Tensor, edge_index, size) -> torch.Tensor:
        # x_j has shape [E, out_channels]-15

        # Step 3: Normalize node features.
        src, dst = edge_index  # we assume source_to_target message passing
        deg = degree(src, size[0], dtype=x_j.dtype)
        deg = deg.pow(-1)
        norm = deg[dst]

        return norm.view(-1, 1) * x_j  # broadcasting the normalization term to all out_channels === hidden features

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


"""def train_step_classifier(model: DGCNN, train_data: DataLoader, optimizer, device: torch.device,
                          criterion: Callable = CrossEntropyLoss()):
    # TODO: test this
    # Put the model in training mode
    model.train()

    # Running average loss over the batches
    running_loss = 0.0
    steps: int = 1

    for data in iter(train_data):
        # move batch to device
        data = data.to(device)
        # reset the optimizer gradients
        optimizer.zero_grad()

        # Encoder output
        y_hat = model(data.x, data.edge_index, data.batch)

        loss = model.loss(y=data.y, y_hat=y_hat, criterion=criterion, additional_terms=None)

        # Gradient update
        loss.backward()
        # Advance the optimizer state
        optimizer.step()

        # Update running average loss
        running_loss = running_loss + 1 / steps * (loss.item() - running_loss)
        print(f"Steps: {steps}/{len(train_data)}, running loss {running_loss}")
        steps += 1

    return float(running_loss)


@torch.no_grad()
def test_step_classifier(model: DGCNN, val_data: DataLoader, device: torch.device,
                         use_edge_weight: bool = False, use_edge_attr: bool = False, top_k: int = 3,
                         criterion: Callable = CrossEntropyLoss()):
    # TODO: test this
    # put the model in evaluation mode
    model.eval()

    # Running average for loss, precision and AUC
    running_val_loss = 0
    running_precision = 0
    running_recall = 0
    running_accuracy = 0
    running_topk_acc = 0
    running_f1 = 0
    steps: int = 1

    for data in iter(val_data):
        # move batch to device
        data = data.to(device)

        if use_edge_weight and use_edge_attr:
            loss, acc, top_k_acc, prec, rec, f1 = model.test(x=data.x, edge_index=data.edge_index, y=data.y,
                                                             batch_index=data.batch, criterion=criterion, top_k=top_k,
                                                             edge_weight=data.edge_weight, edge_attr=data.edge_attr)
        elif use_edge_attr:

            loss, acc, top_k_acc, prec, rec, f1 = model.test(x=data.x, edge_index=data.edge_index, y=data.y,
                                                             batch_index=data.batch, criterion=criterion, top_k=top_k,
                                                             edge_attr=data.edge_attr)
        elif use_edge_weight:
            loss, acc, top_k_acc, prec, rec, f1 = model.test(x=data.x, edge_index=data.edge_index, y=data.y,
                                                             batch_index=data.batch, criterion=criterion, top_k=top_k,
                                                             edge_weight=data.edge_weight)
        else:
            loss, acc, top_k_acc, prec, rec, f1 = model.test(x=data.x, edge_index=data.edge_index, y=data.y,
                                                             batch_index=data.batch, criterion=criterion, top_k=top_k)

        running_val_loss = running_val_loss + 1 / steps * (loss - running_val_loss)
        running_precision = running_precision + 1 / steps * (prec - running_precision)
        running_recall = running_recall + 1 / steps * (rec - running_recall)
        running_accuracy = running_accuracy + 1 / steps * (acc - running_accuracy)
        running_topk_acc = running_topk_acc + 1 / steps * (top_k_acc - running_topk_acc)
        running_f1 = running_f1 + 1 / steps * (f1 - running_f1)

        steps += 1
    return float(running_precision), float(running_recall), float(running_accuracy), float(running_topk_acc), \
        float(running_f1), float(running_val_loss)


def train_dgcnn(model: DGCNN, train_data: DataLoader, val_data: DataLoader, epochs: int, optimizer,
                experiment_path: str, experiment_name: str, early_stopping_patience: int = EARLY_STOP_PATIENCE,
                early_stopping_delta: float = 0, top_k: int = 3, criterion: Callable = CrossEntropyLoss()) -> \
        torch.nn.Module:
    # TODO: test this
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    experiment_path = os.path.join(experiment_path, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)  # create experiment directory if it doesn't exist

    # Instantiate the summary writer
    writer = SummaryWriter(f'{experiment_path}_{experiment_name}_{epochs}_epochs')

    # Early-stopping monitor
    checkpoint_path = os.path.join(f"{experiment_path}", "checkpoint.pt")
    monitor = EarlyStopping(
        patience=early_stopping_patience,
        verbose=True,
        delta=early_stopping_delta,
        path=checkpoint_path
    )

    # Metric history trace object
    mht = MetricsHistoryTracer(
        metrics=[
            "avg_precision",
            "avg_recall",
            "avg_accuracy",
            f"avg_top{top_k}_accuracy",
            "avg_f1",
            "val_loss",
            "train_loss"
        ],
        name="Classifier training metrics"
    )

    for epoch in range(0, epochs):
        # Do train step
        train_loss = train_step_classifier(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            device=device
        )

        # Do validation step
        avg_precision, avg_recall, avg_accuracy, avg_topk_accuracy, avg_f1, val_loss = test_step_classifier(
            model=model,
            val_data=val_data,
            device=device,
            top_k=top_k,
            criterion=criterion
        )

        print(
            'Epoch: {:d}, Train loss: {:.4f}, Validation loss {:.4f}, Average accuracy: {:.4f}, '
            'Average top-{:d} accuracy: {:.4f}, Average precision: {:.4f}, Average recall: {:.4f}, Average F1: {:.4f}, '
            .format(epoch, train_loss, val_loss, avg_accuracy, top_k,
                    avg_topk_accuracy, avg_precision, avg_recall, avg_f1)
        )

        # Tensorboard state update
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('avg_precision', avg_precision, epoch)
        writer.add_scalar('avg_recall', avg_recall, epoch)
        writer.add_scalar('avg_accuracy', avg_accuracy, epoch)
        writer.add_scalar(f'avg_top{top_k}_accuracy', avg_topk_accuracy, epoch)
        writer.add_scalar('avg_f1', avg_f1, epoch)

        # Check for early-stopping stuff
        monitor(val_loss, model)
        if monitor.early_stop:
            print(f"Epoch {epoch}: early stopping, restoring model checkpoint {checkpoint_path}...")
            break

        # Metrics history update
        mht.add_scalar('train_loss', train_loss)
        mht.add_scalar('val_loss', val_loss)
        mht.add_scalar('avg_precision', avg_precision)
        mht.add_scalar('avg_recall', avg_recall)
        mht.add_scalar('avg_accuracy', avg_accuracy)
        mht.add_scalar(f'avg_top{top_k}_accuracy', avg_topk_accuracy)
        mht.add_scalar('avg_f1', avg_f1)

    # Plot the metrics
    mht.plot_metrics(
        [
            'train_loss',
            'val_loss',
        ],
        figsize=FIGURE_SIZE_DEFAULT,
        traced_min_metric='val_loss',
        store_path=os.path.join(f"{experiment_path}", "loss.svg")
    )

    mht.plot_metrics(
        [
            "avg_precision",
            "avg_recall",
            "avg_f1",
        ],
        figsize=FIGURE_SIZE_DEFAULT,
        traced_max_metric='avg_f1',
        store_path=os.path.join(f"{experiment_path}", "prec_rec_f1.svg")
    )

    mht.plot_metrics(
        [
            'avg_accuracy',
        ],
        figsize=FIGURE_SIZE_DEFAULT,
        traced_max_metric='avg_accuracy',
        store_path=os.path.join(f"{experiment_path}", "avg_accuracy.svg")
    )

    mht.plot_metrics(
        [
            f'avg_top{top_k}_accuracy',
        ],
        figsize=FIGURE_SIZE_DEFAULT,
        traced_max_metric=f'avg_top{top_k}_accuracy',
        store_path=os.path.join(f"{experiment_path}", f'avg_top{top_k}_accuracy.svg')
    )

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    return model"""
