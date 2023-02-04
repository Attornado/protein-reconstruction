import os
import torch
from typing import Callable, Type
from torch.nn import LayerNorm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch.utils.data import RandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import DeepGraphInfomax
from models.layers import SerializableModule
from torch_geometric.nn.pool import global_mean_pool
from training.training_tools import FIGURE_SIZE_DEFAULT, MetricsHistoryTracer, EarlyStopping, EARLY_STOP_PATIENCE


# TODO: implement serialization for DGI


def readout_function(encoding: torch.Tensor):
    # must test this
    return torch.sigmoid(
        global_mean_pool(x=encoding, size=1)
    )


def random_sample_corruption(training_set: DataLoader, graph: Data) -> Data:
    """
    Takes a DataLoader and returns a single random sample from it.

    :param graph: graph data batch
    :type graph: Data
    :param training_set: DataLoader of the graph dataset
    :type training_set: DataLoader
    :return: A single batch sample from the training set.
    """
    corrupted_graph = graph
    # Check adj matrix, should be enough
    while corrupted_graph.edge_index.equal(graph.edge_index):
        train_sample = RandomSampler(
            training_set,
            replacement=False,
            num_samples=1,
            generator=None
        )
        corrupted_graph = next(iter(train_sample))
    return corrupted_graph


class DeepGraphInfomaxWrapper(DeepGraphInfomax, SerializableModule):
    # TODO check readout function inside the init
    def __init__(self,
                 hidden_channels: int,
                 encoder: SerializableModule,
                 readout: Callable,
                 corruption: Callable,
                 normalize_hidden: bool = True,
                 dropout: float = 0.0
                 ):
        super().__init__(hidden_channels=hidden_channels, encoder=encoder, summary=readout, corruption=corruption)

        self._norm = None
        self.__hidden_channels = hidden_channels
        self.__normalize_hidden = normalize_hidden
        self.__dropout = dropout

        if normalize_hidden:
            self._norm = LayerNorm(hidden_channels, elementwise_affine=True)

    @property
    def dropout(self) -> float:
        return self.__dropout

    @property
    def hidden_channels(self) -> int:
        return self.__hidden_channels

    @hidden_channels.setter
    def hidden_channels(self, value):
        self.__hidden_channels = value

    @property
    def normalize_hidden(self) -> bool:
        return self.__normalize_hidden

    # noinspection PyTypedDict
    def serialize_constructor_params(self, *args, **kwargs) -> dict:

        constructor_params = {
            "hidden_channels": self.__hidden_channels,
            "dropout": self.__dropout,
            "normalize_hidden": self.__normalize_hidden
        }

        # Serialize encoder
        constructor_params["encoder"] = {
            "state_dict": self.encoder.state_dict(),
            "constructor_params": self.encoder.serialize_constructor_params()
        }

        return constructor_params

    # noinspection PyMethodOverriding
    @classmethod
    def from_constructor_params(cls,
                                constructor_params: dict,
                                encoder_constructor: Type[SerializableModule],
                                readout: Callable,
                                corruption: Callable,
                                *args, **kwargs):
        # Get encoder constructor params/state dict and construct it
        enc_state_dict = constructor_params["encoder"]["state_dict"]
        enc_constructor_params = constructor_params["encoder"]["constructor_params"]
        encoder = encoder_constructor.from_constructor_params(enc_constructor_params)  # construct encoder
        encoder.load_state_dict(state_dict=enc_state_dict)  # set weights

        # Get other params
        hidden_channels = constructor_params["hidden_channels"]
        normalize_hidden = constructor_params["normalize_hidden"]
        dropout = constructor_params["dropout"]

        return cls(
            hidden_channels=hidden_channels,
            encoder=encoder,
            readout=readout,
            corruption=corruption,
            normalize_hidden=normalize_hidden,
            dropout=dropout
        )

    def forward(self, x, edge_index, *args, **kwargs):
        pos_z, neg_z, summary = super().forward(x, edge_index, *args, **kwargs)

        if self._norm is not None:
            summary = self._norm(summary).relu()
        else:
            summary = F.relu(summary)

        # Apply dropout
        if self.__dropout > 0:
            summary = F.dropout(summary, p=self.__dropout, training=self.training)
        return pos_z, neg_z, summary

    def test_discriminator(self, x, edge_index, threshold=0.5, *args, **kwargs):
        """
        Takes in the model, the data, and a threshold value. It then calculates the precision, recall, accuracy, and f1
        score of the model.

        :param x: the node features
        :param edge_index: The edge index of the graph
        :param threshold: The threshold for the discriminator to classify a node as positive or negative
        :return: Precision, recall, accuracy, and f1 score.
        """
        pos_z, neg_z, summary = self(x, edge_index, *args, **kwargs)

        pos = self.discriminate(pos_z, summary, sigmoid=True)
        neg = 1 - self.discriminate(neg_z, summary, sigmoid=True)

        pos_pred = (pos >= threshold).float()
        neg_pred = (neg >= threshold).float()

        true_positive = torch.count_nonzero(pos_pred)
        true_negative = torch.count_nonzero(neg_pred)
        false_positive = neg_pred.shape[-1] - true_negative
        false_negative = pos_pred.shape[-1] - true_positive

        precision = true_positive / (true_positive + false_positive)
        recall = true_negative / (true_positive + false_negative)
        acc = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        f1_score = (2 * precision * recall) / (precision + recall)

        return precision, recall, acc, f1_score


def train_step_DGI(model: DeepGraphInfomaxWrapper, train_data: DataLoader, optimizer, device,
                   use_edge_weight: bool = False, use_edge_attr: bool = False):
    # put the model in training mode
    model.train()

    # running average loss over the batches
    running_loss = 0.0
    steps: int = 1

    for data in iter(train_data):
        # move batch to device
        data = data.to(device)
        # reset the optimizer gradients
        optimizer.zero_grad()

        # Encoder output
        if use_edge_weight and use_edge_attr:
            pos_z, neg_z, summary = model(
                data.x,
                data.edge_index,
                edge_attr=data.edge_attr,
                edge_weight=data.edge_weight
            )
        elif use_edge_attr:
            pos_z, neg_z, summary = model(data.x, data.edge_index, edge_attr=data.edge_attr)
        elif use_edge_weight:
            pos_z, neg_z, summary = model(data.x, data.edge_index, edge_weight=data.edge_weight)
        else:
            pos_z, neg_z, summary = model(data.x, data.edge_index)

        loss = model.loss(pos_z, neg_z, summary)

        # gradient update
        loss.backward()
        # advance the optimizer state
        optimizer.step()

        # Update running average loss
        running_loss = running_loss + 1 / steps * (loss.item() - running_loss)
        steps += 1
    return float(running_loss)


@torch.no_grad()
def test_step_DGI(model: DeepGraphInfomaxWrapper, val_data: DataLoader, device, use_edge_weight: bool = False,
                  use_edge_attr: bool = False, threshold: float = 0.5):
    # put the model in evaluation mode
    model.eval()

    # Running average for loss, precision and AUC
    running_val_loss = 0
    running_precision = 0
    running_recall = 0
    running_accuracy = 0
    running_f1 = 0
    steps: int = 1

    for data in iter(val_data):
        # move batch to device
        data = data.to(device)

        # Encoder output
        if use_edge_weight and use_edge_attr:
            pos_z, neg_z, summary = model(data.x, data.edge_index, edge_attr=data.edge_attr,
                                          edge_weight=data.edge_weight)
        elif use_edge_attr:
            pos_z, neg_z, summary = model(data.x, data.edge_index, edge_attr=data.edge_attr)
        elif use_edge_weight:
            pos_z, neg_z, summary = model(data.x, data.edge_index, edge_weight=data.edge_weight)
        else:
            pos_z, neg_z, summary = model(data.x, data.edge_index)

        loss = model.loss(pos_z, neg_z, summary)
        precision, recall, accuracy, f1 = model.test_discriminator(data.x, data.edge_index, threshold)

        running_val_loss = running_val_loss + 1 / steps * (loss.item() - running_val_loss)
        running_precision = running_precision + 1 / steps * (precision - running_precision)
        running_recall = running_recall + 1 / steps * (recall - running_recall)
        running_accuracy = running_accuracy + 1 / steps * (accuracy - running_accuracy)
        running_f1 = running_f1 + 1 / steps * (f1 - running_f1)

        steps += 1
    return float(running_precision), float(running_recall), float(running_accuracy), float(running_f1), \
        float(running_val_loss)


def train_DGI(model: DeepGraphInfomaxWrapper, train_data: DataLoader, val_data: DataLoader, epochs: int, optimizer,
              experiment_path: str, experiment_name: str, use_edge_weight: bool = False, use_edge_attr: bool = False,
              early_stopping_patience: int = EARLY_STOP_PATIENCE, early_stopping_delta: float = 0,
              threshold: float = 0.5) -> torch.nn.Module:
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

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
            "avg_f1",
            "avg_val_loss",
        ],
        name="DGI training metrics"
    )

    for epoch in range(0, epochs):
        # Do train step
        train_loss = train_step_DGI(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            device=device,
            use_edge_weight=use_edge_weight,
            use_edge_attr=use_edge_attr
        )

        # Do validation step
        avg_precision, avg_recall, avg_accuracy, avg_f1, val_loss = test_step_DGI(
            model=model,
            val_data=val_data,
            device=device,
            use_edge_weight=use_edge_weight,
            use_edge_attr=use_edge_attr,
            threshold=threshold
        )

        print(
            'Epoch: {:d}, Train loss: {:.4f}, Validation loss {:.4f}, ' 'Average: {:.4f}, Average precision: {:.4f}'
            .format(
                epoch,
                train_loss,
                val_loss,
                avg_precision,
                avg_recall,
                avg_accuracy,
                avg_f1
            )
        )

        # Tensorboard state update
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('avg_precision', avg_precision, epoch)
        writer.add_scalar('avg_recall', avg_recall, epoch)
        writer.add_scalar('avg_accuracy', avg_accuracy, epoch)
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
        traced_min_metric='avg_f1',
        store_path=os.path.join(f"{experiment_path}", "prec_rec_f1.svg")
    )

    mht.plot_metrics(
        [
            'avg_accuracy',
        ],
        figsize=FIGURE_SIZE_DEFAULT,
        traced_min_metric='avg_accuracy',
        store_path=os.path.join(f"{experiment_path}", "avg_accuracy.svg")
    )

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    return model
