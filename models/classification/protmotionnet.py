import os
from typing import final, Callable, Type, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.nn.aggr import LSTMAggregation, SoftmaxAggregation, MaxAggregation, MeanAggregation, SumAggregation
from torch_geometric.nn.dense import Linear  # , dense_diff_pool this has to be in the "future work"
from models.layers import SerializableModule
from functools import partial
from torchmetrics.functional import accuracy, f1_score, precision, recall
from training.training_tools import FIGURE_SIZE_DEFAULT, EarlyStopping, MetricsHistoryTracer, EARLY_STOP_PATIENCE


CLASSES: final = 7


class ProtMotionNet(SerializableModule):
    __READOUTS: final = {
        "mean_pool": MeanAggregation(),
        "max_pool": MaxAggregation(),
        "add_pool": SumAggregation(),
        'lstm': LSTMAggregation,
        'softmax': SoftmaxAggregation(learn=True)
    }
    READOUTS: final = frozenset(__READOUTS.keys())

    __ACTIVATIONS: final = {
        "linear": F.linear,
        "relu": F.relu,
        "leaky_relu": F.leaky_relu,
        "rrelu": F.rrelu,
        "relu6": F.relu6,
        "gelu": F.gelu,
        "elu": F.elu,
        "celu": F.celu,
        "glu": F.glu,
        "selu": F.selu,
        "prelu": F.prelu,
        "silu": F.silu,
        "hardswish": F.hardswish,
        "tanh": F.tanh,
        "sigmoid": torch.sigmoid,
        "softmax": partial(F.softmax, dim=-1)
    }
    ACTIVATIONS: final = frozenset(__ACTIVATIONS.keys())

    def __init__(self, encoder: SerializableModule, encoder_out_channels: int, dense_units: list[int],
                 dense_activations: list[str], dropout: float = 0.0, readout: str = 'mean_pool'):

        super().__init__()

        # Validate input
        if len(dense_units) != len(dense_activations):
            raise ValueError(f"len(dense_activations) must be equal to len(dense_units),"
                             f" {len(dense_activations)} and {len(dense_units)} given")

        self._encoder = encoder
        self.__encoder_out_channels = encoder_out_channels
        self.__dense_units = dense_units
        self.__dense_activations = dense_activations
        self.__dropout = dropout
        self.__readout = readout
        self._readout_aggregation = self.__resolve_readout(readout, encoder_out_channels)

        # Build dense layers
        self._dense_layers = torch.nn.ModuleList()
        prev_units = encoder_out_channels
        for i in range(0, len(dense_units)):

            # Check if activations
            if dense_activations[i] not in self.__ACTIVATIONS:
                raise ValueError(f"Activation function must be one of {self.ACTIVATIONS}, {dense_activations[i]} given")

            # Build dense layer
            self._dense_layers.append(Linear(prev_units, dense_units[i]))
            prev_units = dense_units[i]

    def forward(self, x, edge_index, batch_index: Tensor = None, *args, **kwargs):

        # Extract features with encoder
        x = self._encoder(x, edge_index, *args, **kwargs)

        # Apply readout aggregation, assuming batch is a single graph if batch_index is not given
        if batch_index is None:
            batch_index = torch.zeros(size=(x.shape[-2],)).type(torch.int64)
        x = self._readout_aggregation(x, index=batch_index)

        for i in range(0, len(self._dense_layers)):

            # Apply dense layer and activation
            dense = self._dense_layers[i]
            activation = self.__ACTIVATIONS[self.__dense_activations[i]]
            x = dense(x)
            x = activation(x)

            # Apply dropout on the activation, except if it is the output one
            if i < len(self._dense_layers) - 1:
                x = F.dropout(x, p=self.__dropout, training=self.training)

        return x

    def serialize_constructor_params(self, *args, **kwargs) -> dict:

        # Serialize constructor parameters
        constructor_params = {
            "encoder_out_channels": self.__encoder_out_channels,
            "dense_units": self.__dense_units,
            "dense_activations": self.__dense_activations,
            "dropout": self.__dropout,
            "readout": self.__readout
        }

        # Serialize encoder
        constructor_params["encoder"] = {
            "constructor_params": self._encoder.serialize_constructor_params(),
            "state_dict": self._encoder.state_dict()
        }

        return constructor_params

    # noinspection PyMethodOverriding
    @classmethod
    def from_constructor_params(cls,
                                constructor_params: dict,
                                encoder_constructor: Type[SerializableModule],
                                *args, **kwargs):
        # Deserialize encoder
        encoder_constructor_params = constructor_params["encoder"]["constructor_params"]
        encoder_state_dict = constructor_params["encoder"]["state_dict"]
        # TODO: Could not work if from_constructor_params is overridden by encoder class, must fix this to generalize
        encoder = encoder_constructor.from_constructor_params(encoder_constructor_params)
        encoder.load_state_dict(encoder_state_dict)
        del constructor_params["encoder"]  # delete encoder params from constructor_params

        return cls(encoder=encoder, **constructor_params)

    @classmethod
    def __resolve_readout(cls, readout: str, channels: int) -> Callable:
        if readout not in cls.READOUTS:
            raise ValueError(f"Readout function must be in {cls.READOUTS}")

        if readout == 'lstm':
            return LSTMAggregation(in_channels=channels, out_channels=channels)
        return cls.__READOUTS[readout]

    def test(self, x: torch.Tensor, edge_index: torch.Tensor, y, batch_index: torch.Tensor = None,
             criterion: Callable = CrossEntropyLoss(), top_k: Optional[int] = None, *args, **kwargs) -> \
            (float, Optional[float], float, float, float, float):
        """
        This function takes in a graph, and returns the loss, accuracy, top-k accuracy, precision, recall, and F1-score.

        :param x: torch.Tensor = The node features
        :type x: torch.Tensor
        :param edge_index: The edge indices of the graph
        :type edge_index: torch.Tensor
        :param y: The target labels
        :param batch_index: The batch index of the nodes
        :type batch_index: torch.Tensor
        :param criterion: The loss function to use
        :type criterion: Callable
        :param top_k: k for computing top_k accuracy, *args, **kwargs
        :type top_k: Optional[int]
        :return: The loss, accuracy, top-k accuracy, precision, recall, and F1-score.
        """

        # TODO: test this

        # Get the number of classes
        n_classes = self.__dense_units[-1]

        # Get predictions
        y_hat = self(x, edge_index, batch_index, *args, **kwargs)

        # Compute loss
        loss = self.loss(y_hat=y_hat, y=y, criterion=criterion)

        # Compute the metrics
        acc = accuracy(preds=y_hat, target=y, task='multiclass', num_classes=n_classes)
        if top_k is not None:
            top_k_acc = float(accuracy(preds=y_hat, target=y, task='multiclass', num_classes=n_classes, top_k=top_k))
        else:
            top_k_acc = None
        prec = precision(preds=y_hat, target=y, task='multiclass', num_classes=n_classes, average="weighted")
        rec = recall(preds=y_hat, target=y, task='multiclass', num_classes=n_classes, average="weighted")
        f1 = f1_score(preds=y_hat, target=y, task='multiclass', num_classes=n_classes, average="weighted")

        return float(loss), float(acc), top_k_acc, prec, rec, f1

    def loss(self, y, x: Optional[torch.Tensor] = None, edge_index: Optional[torch.Tensor] = None,
             batch_index: Optional[torch.Tensor] = None, y_hat: Optional[torch.Tensor] = None,
             criterion: Callable = CrossEntropyLoss(), additional_terms: list[Tensor] = None, *args, **kwargs) -> \
            torch.Tensor:
        # TODO: test this

        # If predictions are not given, compute them using the model
        if y_hat is None:
            y_hat = self(x, edge_index, batch_index, *args, **kwargs)

        # Compute loss with given criterion
        loss = criterion(y_hat, y)

        # Add pre-computed additional loss terms to the loss
        if additional_terms is not None:
            for additional_term in additional_terms:
                loss = loss + additional_term

        return loss


def train_step_classifier(model: ProtMotionNet, train_data: DataLoader, optimizer, device: torch.device,
                          use_edge_weight: bool = False, use_edge_attr: bool = False,
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
        if use_edge_weight and use_edge_attr:
            y_hat = model(data.x, data.edge_index, batch_index=data.batch,
                          edge_attr=data.edge_attr, edge_weight=data.edge_weight)
        elif use_edge_attr:
            y_hat = model(data.x, data.edge_index, batch_index=data.batch, edge_attr=data.edge_attr)
        elif use_edge_weight:
            y_hat = model(data.x, data.edge_index, batch_index=data.batch, edge_weight=data.edge_weight)
        else:
            y_hat = model(data.x, data.edge_index, batch_index=data.batch)

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
def test_step_classifier(model: ProtMotionNet, val_data: DataLoader, device: torch.device,
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


def train_classifier(model: ProtMotionNet, train_data: DataLoader, val_data: DataLoader, epochs: int, optimizer,
                     experiment_path: str, experiment_name: str, use_edge_weight: bool = False,
                     use_edge_attr: bool = False,
                     early_stopping_patience: int = EARLY_STOP_PATIENCE, early_stopping_delta: float = 0,
                     top_k: int = 3, criterion: Callable = CrossEntropyLoss()) -> torch.nn.Module:
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
            device=device,
            use_edge_weight=use_edge_weight,
            use_edge_attr=use_edge_attr
        )

        # Do validation step
        avg_precision, avg_recall, avg_accuracy, avg_topk_accuracy, avg_f1, val_loss = test_step_classifier(
            model=model,
            val_data=val_data,
            device=device,
            use_edge_weight=use_edge_weight,
            use_edge_attr=use_edge_attr,
            top_k=top_k,
            criterion=criterion
        )

        print(
            'Epoch: {:d}, Train loss: {:.4f}, Validation loss {:.4f}, Average accuracy: {:.4f}, '
            'Average top-{:d} accuracy: {:.4f}, Average precision: {:.4f}, Average recall: {:.4f}, Average F1: {:.4f}, '
            .format(epoch + 1, train_loss, val_loss, avg_accuracy, top_k,
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
    return model
