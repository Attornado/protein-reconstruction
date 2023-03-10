import abc
import os
import torch
from typing import Callable, Type, Optional
from torch.nn import LayerNorm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import DeepGraphInfomax
from models.layers import SerializableModule
from torch_geometric.nn.pool import global_mean_pool
from training.training_tools import FIGURE_SIZE_DEFAULT, MetricsHistoryTracer, EarlyStopping, EARLY_STOP_PATIENCE


class MeanPoolReadout(object):
    def __init__(self, device: Optional[torch.device] = None, sigmoid: bool = False):
        """
        Represents a mean pooling readout function that tracks the last batch of the given Data.

        :param device: The device to use for the computation.
        :param sigmoid: If True, the output will be passed through a sigmoid activation function, defaults to False
        :type sigmoid: bool (optional)
        """
        self.__sigmoid: bool = sigmoid
        self.__batch: Optional[torch.Tensor] = None  # last batch
        if device is not None:
            self.__device: Optional[torch.device] = device
        else:
            self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, encoding: torch.Tensor, x: torch.Tensor, edge_index,
                 batch: Optional[torch.Tensor] = None, *args, **kwargs):
        if batch is None and self.__batch is not None:
            batch = self.__batch
        elif batch is None:
            # Assume it's a single batch if not given
            batch = torch.zeros((encoding.shape[-2])).type(torch.int64).to(self.__device)

        # Update the batch in any case
        self.__batch = batch

        if self.__sigmoid:
            return torch.sigmoid(
                global_mean_pool(x=encoding, batch=batch, size=None)
            )
        else:
            return global_mean_pool(x=encoding, batch=batch, size=None)

    @property
    def sigmoid(self) -> bool:
        return self.__sigmoid

    @sigmoid.setter
    def sigmoid(self, sigmoid: bool):
        self.__sigmoid = sigmoid

    @property
    def batch(self) -> Optional[torch.Tensor]:
        return self.__batch

    @batch.setter
    def batch(self, batch: Optional[torch.Tensor]):
        self.__batch = batch

    @property
    def device(self) -> Optional[torch.device]:
        return self.__device

    @device.setter
    def device(self, device: Optional[torch.device]):
        self.__device = device


class CorruptionFunction(abc.ABC):
    def __init__(self, device: torch.device):
        """
        This class represents a generic corruption function to be used in Deep Graph Infomax models.

        :param device: The device to run the corruption function on
        :type device: torch.device
        """
        super().__init__()
        self.__device = device
        self.__batch: Optional[torch.Tensor] = None  # last corrupted batch

    @property
    def batch(self) -> Optional[torch.Tensor]:
        """
        Returns the batch index tensor the last of corrupted data if it exists, None otherwise.

        rtype: Optional[torch.Tensor]
        """
        return self.__batch

    @batch.setter
    def batch(self, batch: Optional[torch.Tensor]):
        """
        Sets the last batch index tensor.

        :param batch: last batch index of the corrupted data.
        :type batch: Optional[torch.Tensor]
        """
        self.__batch = batch

    @property
    def device(self) -> torch.device:
        """
        Returns the corruption function device.

        :rtype: torch.device
        """
        return self.__device

    @device.setter
    def device(self, device: torch.device):
        """
        Sets the device the corruption function is running on.

        :param device: The device to run the corruption on
        :type device: torch.device
        """
        self.__device = device

    def to(self, device: torch.device):
        """
        Sets the corruption function device to the given one.

        :param device: The device to run the corruption on
        :type device: torch.device
        """
        self.device = device

    def __call__(self, x: torch.Tensor, edge_index: torch.Tensor, return_batch: bool = False,
                 batch: Optional[torch.Tensor] = None, *args, **kwargs):
        """
        Takes graph signal x and its edge index, returning a corrupted graph.

        :param x: node feature signal
        :type x: torch.Tensor
        :param edge_index: The edge indices of the graph
        :type edge_index: torch.Tensor
        :param return_batch: If True, the output will be a batch of graphs. If False, the output will not contain the
            batch information.
        :type return_batch: bool (optional)
        :param batch: batch index of the input data
        :type batch: Optional[torch.Tensor] (optional)
        """
        raise NotImplementedError(f"__call__ method must be implemented in a {self.__class__.__name__} subclass.")


class RandomPermutationCorruption(CorruptionFunction):
    def __init__(self, device: torch.device):
        """
        A corruption function that randomly permutes the given data batch features.

        :param device: The device to run the corruption on
        :type device: torch.device
        """
        super().__init__(device)

    def __call__(self, x: torch.Tensor, edge_index: torch.Tensor, return_batch: bool = False,
                 batch: Optional[torch.Tensor] = None, *args, **kwargs):
        """
        Takes graph or batch graph signal x and its edge index, returning a corrupted graph obtained permuting the
        nodes .

        :param x: node feature signal
        :type x: torch.Tensor
        :param edge_index: The edge indices of the graph
        :type edge_index: torch.Tensor
        :param return_batch: If True, the output will be a batch of graphs. If False, the output will not contain the
            batch information.
        :type return_batch: bool (optional)
        :param batch: batch index of the input data
        :type batch: Optional[torch.Tensor] (optional)
        :return the corrupted graph
        """

        # Permute node features
        corrupted_x = x[torch.randperm(x.shape[0])]

        # Maybe permute some edges too with torch_geometric.utils.negative_sampling.batched_negative_sampling()?

        return_tuple = [corrupted_x, edge_index]

        # Get additional features from kwargs
        for k in kwargs:
            return_tuple.append(kwargs[k])  # add other required graph features

        # Batch remains the same
        self.batch = batch
        if return_batch:
            return_tuple.append(batch)

        return tuple(return_tuple)


class RandomSampleCorruption(CorruptionFunction):
    def __init__(self, train_data: DataLoader, val_data: DataLoader, device: torch.device):
        super().__init__(device)
        self.__train_data = train_data
        self.__val_data = val_data
        self.__iter_train_data = iter(train_data)
        self.__iter_val_data = iter(val_data)
        self.__device = device
        self.__training = True

    def __call__(self, x: torch.Tensor, edge_index: torch.Tensor, return_batch: bool = False,
                 batch: Optional[torch.Tensor] = None, *args, **kwargs):
        # Loop until we don't find a different batch of graphs comparing the edges
        corrupted_edges = edge_index
        corrupted_graph = None
        while corrupted_edges.equal(edge_index):

            # Get the data loader of the train if we are in training mode and the validation one if we are in the test
            iter_loader = self.__iter_train_data if self.__training else self.__iter_val_data

            # Try to get next item
            try:
                corrupted_graph = next(iter_loader)

            # On data loader stop, reset it
            except StopIteration:
                if self.__training:
                    self.__iter_train_data = iter(self.__train_data)
                    corrupted_graph = next(self.__iter_train_data)
                else:
                    self.__iter_val_data = iter(self.__val_data)
                    corrupted_graph = next(self.__iter_val_data)

            corrupted_graph = corrupted_graph.to(self.__device)
            corrupted_edges = corrupted_graph.edge_index

        return_tuple = [corrupted_graph.x, corrupted_graph.edge_index]

        for k in kwargs:
            if k in corrupted_graph:
                return_tuple.append(corrupted_graph[k])  # add other required graph features

        self.batch = corrupted_graph.batch
        if return_batch:
            return_tuple.append(corrupted_graph.batch)

        return tuple(return_tuple)

    @property
    def training(self) -> bool:
        return self.__training

    @training.setter
    def training(self, train: bool):
        self.__training = train

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class DeepGraphInfomaxV2(DeepGraphInfomax, SerializableModule):
    def __init__(self,
                 hidden_channels: int,
                 encoder: SerializableModule,
                 readout: Callable,
                 corruption: CorruptionFunction,
                 normalize_hidden: bool = True,
                 dropout: float = 0.0
                 ):
        super().__init__(hidden_channels=hidden_channels, encoder=encoder, summary=readout, corruption=corruption)

        self._norm = None
        self.__normalize_hidden = normalize_hidden
        self.__dropout = dropout

        if normalize_hidden:
            self._norm = LayerNorm(hidden_channels, elementwise_affine=True)

    @property
    def dropout(self) -> float:
        return self.__dropout

    @property
    def normalize_hidden(self) -> bool:
        return self.__normalize_hidden

    # noinspection PyTypedDict
    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        constructor_params = {
            "hidden_channels": self.hidden_channels,
            "dropout": self.__dropout,
            "normalize_hidden": self.__normalize_hidden
        }

        # Serialize encoder
        constructor_params["encoder"] = {
            "state_dict": self.encoder.state_dict(),
            "constructor_params": self.encoder.serialize_constructor_params()
        }

        return constructor_params

    def train(self, mode: bool = True):
        super().train(mode)
        if isinstance(self.corruption, RandomSampleCorruption):
            self.corruption.train()

    def eval(self):
        super().eval()
        if isinstance(self.corruption, RandomSampleCorruption):
            self.corruption.eval()

    # noinspection PyMethodOverriding
    @classmethod
    def from_constructor_params(cls,
                                constructor_params: dict,
                                encoder_constructor: Type[SerializableModule],
                                readout: Callable,
                                corruption: CorruptionFunction,
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

    def forward(self, x, edge_index, batch=None, *args, **kwargs):
        # Setup batch if readout requires it
        if isinstance(self.summary, MeanPoolReadout):
            self.summary.batch = batch

        pos_z, neg_z, summary = super().forward(x=x, edge_index=edge_index, *args, **kwargs)

        if self._norm is not None:
            summary = self._norm(summary)

        # Apply dropout
        if self.__dropout > 0:
            summary = F.dropout(summary, p=self.__dropout, training=self.training)
        return pos_z, neg_z, summary

    def test_discriminator(self, x, edge_index, batch: Optional[torch.Tensor] = None, threshold=0.5, *args, **kwargs):
        """
        Takes in the model, the data, and a threshold value. It then calculates the precision, recall, accuracy, and f1
        score of the model.

        :param x: the node features
        :param edge_index: The edge index of the graph data
        :param batch: batch index of the given graph data
        :param threshold: The threshold for the discriminator to classify a node as positive or negative
        :return: Precision, recall, accuracy, and f1 score.
        """
        # pos_z, neg_z, summary = self(x, edge_index, batch=batch, *args, **kwargs)

        # Get positive sample predictions and the corresponding summary
        pos_z = self.encoder(x, edge_index, *args, **kwargs)
        summary = self.summary(pos_z, x, edge_index, batch=batch, *args, **kwargs)

        # Get corrupted graphs and corresponding batch index
        neg_batch = None
        if batch is not None:
            cor = self.corruption(x, edge_index, return_batch=True, batch=batch, *args, **kwargs)
            neg_batch = cor[-1]  # get the negative sample batching
            cor = cor[0:-1]  # remove the batch
        else:
            cor = self.corruption(x, edge_index, *args, **kwargs)

        cor = cor if isinstance(cor, tuple) else (cor, )
        neg_z = self.encoder(*cor)

        # Get positive and negative predictions
        pos = self.discriminate(pos_z, summary, sigmoid=True)
        neg = 1 - self.discriminate(neg_z, summary, sigmoid=True)

        # Slice predictions by batch
        pos_aggregated = pos
        neg_aggregated = neg
        if batch is not None:
            # pos_transpose = torch.transpose(pos, dim0=-1, dim1=0)
            # neg_transpose = torch.transpose(neg, dim0=-1, dim1=0)
            pos_aggregated = []
            neg_aggregated = []

            # For each node in the positive batch
            for i in range(0, batch.shape[0]):

                # Get the graph of the node
                graph_index = batch[i]

                # Get the predictions for the corresponding graph
                pos_aggregated.append(float(pos[i][graph_index]))

            # For each node in the negative batch
            for i in range(0, neg_batch.shape[0]):

                # Get the graph of the node
                graph_index = neg_batch[i]

                # Get the predictions for the corresponding graph
                neg_aggregated.append(float(neg[i][graph_index]))

            # Convert to tensor
            pos_aggregated = torch.tensor(pos_aggregated, device=pos.device)
            neg_aggregated = torch.tensor(neg_aggregated, device=pos.device)

        pos_pred = (pos_aggregated >= threshold).float()
        neg_pred = (neg_aggregated >= threshold).float()

        true_positive = torch.count_nonzero(pos_pred)
        true_negative = torch.count_nonzero(neg_pred)
        false_positive = pos_pred.shape[0] - true_positive
        false_negative = neg_pred.shape[0] - true_negative

        if float(true_positive) + float(false_positive) == 0:
            precision = 0
        else:
            precision = true_positive / (true_positive + false_positive)
        if float(true_positive) + float(false_negative) == 0:
            recall = 0
        else:
            recall = true_positive / (true_positive + false_negative)

        if float(true_positive) + float(false_negative) + float(true_negative) + float(false_positive) == 0:
            acc = 0
        else:
            acc = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

        if float(precision) + float(recall) == 0:
            f1_score = 0
        else:
            f1_score = (2 * precision * recall) / (precision + recall)

        return precision, recall, acc, f1_score


def train_step_DGI(model: DeepGraphInfomaxV2, train_data: DataLoader, optimizer, device,
                   use_edge_weight: bool = False, use_edge_attr: bool = False):
    # Put the model in training mode
    model.train()

    # Running average loss over the batches
    running_loss = 0.0
    steps: int = 1

    for data in iter(train_data):
        # Move batch to device
        data = data.to(device)
        # Reset the optimizer gradients
        optimizer.zero_grad()

        # Encoder output
        if use_edge_weight and use_edge_attr:
            pos_z, neg_z, summary = model(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr,
                                          edge_weight=data.edge_weight)
        elif use_edge_attr:
            pos_z, neg_z, summary = model(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        elif use_edge_weight:
            pos_z, neg_z, summary = model(data.x, data.edge_index, batch=data.batch, edge_weight=data.edge_weight)
        else:
            pos_z, neg_z, summary = model(data.x, data.edge_index, batch=data.batch)

        loss = model.loss(pos_z, neg_z, summary)

        # gradient update
        loss.backward()
        # advance the optimizer state
        optimizer.step()

        # Update running average loss
        running_loss = running_loss + 1 / steps * (loss.item() - running_loss)
        print(f"Steps: {steps}/{len(train_data)}, running loss {running_loss}")
        steps += 1

    return float(running_loss)


@torch.no_grad()
def test_step_DGI(model: DeepGraphInfomaxV2, val_data: DataLoader, device, use_edge_weight: bool = False,
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
            pos_z, neg_z, summary = model(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr,
                                          edge_weight=data.edge_weight)
        elif use_edge_attr:
            pos_z, neg_z, summary = model(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        elif use_edge_weight:
            pos_z, neg_z, summary = model(data.x, data.edge_index, batch=data.batch, edge_weight=data.edge_weight)
        else:
            pos_z, neg_z, summary = model(data.x, data.edge_index, batch=data.batch)

        loss = model.loss(pos_z, neg_z, summary)
        precision, recall, accuracy, f1 = model.test_discriminator(data.x, data.edge_index,
                                                                   batch=data.batch, threshold=threshold)

        running_val_loss = running_val_loss + 1 / steps * (loss.item() - running_val_loss)
        running_precision = running_precision + 1 / steps * (precision - running_precision)
        running_recall = running_recall + 1 / steps * (recall - running_recall)
        running_accuracy = running_accuracy + 1 / steps * (accuracy - running_accuracy)
        running_f1 = running_f1 + 1 / steps * (f1 - running_f1)

        steps += 1
    return float(running_precision), float(running_recall), float(running_accuracy), float(running_f1), \
        float(running_val_loss)


def train_DGI(model: DeepGraphInfomaxV2, train_data: DataLoader, val_data: DataLoader, epochs: int, optimizer,
              experiment_path: str, experiment_name: str, use_edge_weight: bool = False, use_edge_attr: bool = False,
              early_stopping_patience: int = EARLY_STOP_PATIENCE, early_stopping_delta: float = 0,
              threshold: float = 0.5) -> torch.nn.Module:
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
            "avg_f1",
            "val_loss",
            "train_loss"
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
            'Epoch: {:d}, Train loss: {:.4f}, Validation loss {:.4f}, '
            'Average accuracy: {:.4f}, Average precision: {:.4f}, Average recall: {:.4f}, Average F1: {:.4f}, '
            .format(
                epoch + 1,
                train_loss,
                val_loss,
                avg_accuracy,
                avg_precision,
                avg_recall,
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

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    return model
