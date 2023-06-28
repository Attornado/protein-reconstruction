import os
from functools import partial
from typing import final, Callable, Union, Optional, Iterable
import torch
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.aggr import MeanAggregation, MaxAggregation, SumAggregation, LSTMAggregation, SoftmaxAggregation
from torchmetrics.functional import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, \
    pearson_corrcoef, concordance_corrcoef
import einops
from log.logger import Logger
from models.layers import SerializableModule
import torch.nn.functional as F
from models.classification.sage import SAGEClassifier
from models.classification.diffpool import DiffPoolEmbedding
from models.pretraining.gunet import GraphUNetV2
from preprocessing.utils import FrozenDict
from training.training_tools import FIGURE_SIZE_DEFAULT, EarlyStopping, MetricsHistoryTracer, EARLY_STOP_PATIENCE, \
    VAL_LOSS_METRIC
from preprocessing.dataset.dataset_creation import NM_EIGENVALUES


DIFF_POOL: final = "diff_pool"
SAGE: final = "sage_c"
GUNET: final = "gunet"
ENCODER_TYPES: final = frozenset([DIFF_POOL, SAGE, GUNET])
N_EIGENVALUES_DEFAULT: final = 6
LOSS: final = VAL_LOSS_METRIC
MSE: final = "mse"
RMSE: final = "rmse"
MAE: final = "mae"
MAPE: final = "mape"
PEARSON: final = "pearson"
CONCORDANCE: final = "concordance"
METRICS_DICT: final = FrozenDict({
    MSE: mean_squared_error,
    RMSE: lambda preds, target: torch.sqrt(mean_squared_error(preds=preds, target=target)),
    MAE: mean_absolute_error,
    MAPE: mean_absolute_percentage_error,
    PEARSON: lambda preds, target: pearson_corrcoef(preds=preds, target=target).mean(),
    CONCORDANCE: lambda preds, target: concordance_corrcoef(preds=preds, target=target).mean()
})


class EigenValueNMNet(SerializableModule):
    __READOUTS: final = {
        "mean_pool": MeanAggregation(),
        "max_pool": MaxAggregation(),
        "add_pool": SumAggregation(),
        'lstm': LSTMAggregation,
        'softmax': SoftmaxAggregation(learn=True)
    }
    READOUTS: final = frozenset(__READOUTS.keys())

    _ACTIVATIONS: final = {
        "linear": torch.nn.Identity(),
        "relu": F.relu,
        "leaky_relu": F.leaky_relu,
        "rrelu": F.rrelu,
        "relu6": F.relu6,
        "gelu": partial(F.gelu, approximate='none'),
        "elu": F.elu,
        "celu": F.celu,
        "glu": F.glu,
        "selu": F.selu,
        "prelu": F.prelu,
        "silu": F.silu,
        "hardswish": F.hardswish,
        "tanh": F.tanh,
        "sigmoid": torch.sigmoid
    }
    ACTIVATIONS: final = frozenset(_ACTIVATIONS.keys())

    def __init__(self,
                 in_channels: int,
                 encoder_out_channels: int,
                 dense_units: list[int],
                 dense_activations: list[str],
                 encoder_type: str = SAGE,
                 dropout: float = 0.0,
                 readout: str = 'mean_pool',  # no effect if encoder type is "diff_pool"
                 n_eigenvalues: int = N_EIGENVALUES_DEFAULT,
                 **encoder_params):

        super().__init__()

        # Validate input
        if len(dense_units) != len(dense_activations):
            raise ValueError(f"len(dense_activations) must be equal to len(dense_units),"
                             f" {len(dense_activations)} and {len(dense_units)} given")

        self.__encoder_out_channels: int = encoder_out_channels
        self.__in_channels = in_channels
        self.__encoder_type = encoder_type
        self.__dense_units = dense_units
        self.__dense_activations = dense_activations
        self.__dropout = dropout
        self.__readout = readout
        self.__encoder_params = encoder_params
        self.__n_eigenvalues = n_eigenvalues
        self._readout_aggregation = self.__resolve_readout(readout, encoder_out_channels)

        # Build encoder
        if encoder_type == DIFF_POOL:
            encoder = DiffPoolEmbedding(dim_features=in_channels, dim_target=in_channels, config=encoder_params)
        elif encoder_type == SAGE:
            encoder_params.update({"return_embeddings": True})
            encoder = SAGEClassifier(dim_features=in_channels, dim_target=in_channels, config=encoder_params)
        elif encoder_type == GUNET:
            encoder = GraphUNetV2(in_channels=in_channels, hidden_channels=encoder_out_channels,
                                  out_channels=encoder_out_channels, **encoder_params)
        else:
            raise ValueError(f"encoder_type must be one of {ENCODER_TYPES}. {encoder_type} given.")
        self._encoder: SerializableModule = encoder

        # Build dense layers
        self._dense_layers = torch.nn.ModuleList()
        prev_units = encoder_out_channels
        dense_units.append(n_eigenvalues)
        dense_activations.append("linear")
        for i in range(0, len(dense_activations)):

            # Check if activations
            if dense_activations[i] not in self._ACTIVATIONS:
                raise ValueError(f"Activation function must be one of {self.ACTIVATIONS}, {dense_activations[i]} given")

            # Build dense layer
            self._dense_layers.append(Linear(prev_units, dense_units[i]))
            prev_units = dense_units[i]

    @property
    def dropout(self) -> float:
        return self.__dropout

    @dropout.setter
    def dropout(self, dropout: float):
        self.__dropout = dropout

    @property
    def dense_units(self) -> list[int]:
        return list(self.__dense_units)

    @property
    def dense_activations(self) -> list[str]:
        return self.__dense_activations

    @property
    def readout(self) -> str:
        return self.__readout

    @property
    def encoder_out_channels(self) -> int:
        return self.__encoder_out_channels

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    @property
    def encoder_type(self) -> str:
        return self.__encoder_type

    @property
    def encoder_params(self) -> dict:
        return self.__encoder_params

    @property
    def n_eigenvalues(self) -> int:
        return self.__n_eigenvalues

    def serialize_constructor_params(self, *args, **kwargs) -> dict:

        # Serialize constructor parameters
        constructor_params = {
            "in_channels": self.in_channels,
            "encoder_out_channels": self.encoder_out_channels,
            "dense_units": self.dense_units,
            "dense_activations": self.dense_activations,
            "encoder_type": self.encoder_type,
            "dropout": self.encoder_type,
            "readout": self.readout,
            "n_eigenvalues": self.n_eigenvalues
        }
        constructor_params.update(self.encoder_params)

        return constructor_params

    @classmethod
    def __resolve_readout(cls, readout: str, channels: int) -> Callable:
        if readout not in cls.READOUTS:
            raise ValueError(f"Readout function must be in {cls.READOUTS}")

        if readout == 'lstm':
            return LSTMAggregation(in_channels=channels, out_channels=channels)
        return cls.__READOUTS[readout]

    def _apply_dense_layers(self, x: torch.Tensor) -> torch.Tensor:

        for i in range(0, len(self._dense_layers)):

            # Apply dense layer and activation
            dense = self._dense_layers[i]

            x = dense(x)

            # Apply activation in all layers except the last one
            if i < len(self._dense_layers) - 1:
                activation = self._ACTIVATIONS[self.dense_activations[i]]
                x = activation(x)

            # Apply dropout on the activation, except if it is the output one
            if i < len(self._dense_layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch_index: Optional[torch.Tensor] = None,
                *args, **kwargs) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        # Apply readout aggregation, assuming batch is a single graph if batch_index is not given
        if batch_index is None:
            batch_index = torch.zeros(size=(x.shape[-2],)).type(torch.int64)

        # Extract features with encoder
        lpl, el = 0, 0
        if self.encoder_type == DIFF_POOL:
            x, lpl, el = self._encoder(x, edge_index, batch_index, *args, **kwargs)
        else:
            x = self._encoder(x, edge_index, batch_index, *args, **kwargs)

        if self.encoder_type != DIFF_POOL:
            x = self._readout_aggregation(x, index=batch_index)

        x = self._apply_dense_layers(x)

        if self.encoder_type == DIFF_POOL:
            return x, lpl, el

        return x

    def loss(self,
             eigenvalues_true: torch.Tensor,
             eigenvalues_pred: Optional[torch.Tensor] = None,
             x: Optional[torch.Tensor] = None,
             edge_index: Optional[torch.Tensor] = None,
             batch_index: Optional[torch.Tensor] = None,
             criterion: Callable = MSELoss()) -> torch.Tensor:
        if eigenvalues_pred is None and (x is None or edge_index is None):
            raise ValueError("Either eigenvalues_pred or x and edge_index must be given.")

        lpl = 0
        el = 0
        eigenvalues_pred_tmp = None
        # Get DiffPool additional loss terms
        if self.encoder_type == DIFF_POOL:
            eigenvalues_pred_tmp, lpl, el = self(x=x, edge_index=edge_index, batch_index=batch_index)

        # Get predictions if not given
        if eigenvalues_pred is None:
            if self.encoder_type == DIFF_POOL:
                eigenvalues_pred = eigenvalues_pred_tmp
            else:
                eigenvalues_pred = self(x=x, edge_index=edge_index, batch_index=batch_index)

        # Compute MSE loss
        loss = criterion(eigenvalues_pred, eigenvalues_true)

        # Add additional DiffPool terms if required
        if self.encoder_type == DIFF_POOL:
            loss = loss + lpl + el
            del eigenvalues_pred_tmp, lpl, el
            torch.cuda.empty_cache()

        del eigenvalues_pred
        torch.cuda.empty_cache()

        return loss

    def test(self,
             eigenvalues_true: torch.Tensor,
             eigenvalues_pred: Optional[torch.Tensor] = None,
             x: Optional[torch.Tensor] = None,
             edge_index: Optional[torch.Tensor] = None,
             batch_index: Optional[torch.Tensor] = None,
             criterion: Callable = MSELoss(),
             metrics: Iterable[str] = frozenset([MAE, MAPE, RMSE])) -> dict[str, float]:
        if eigenvalues_pred is None and (x is None or edge_index is None):
            raise ValueError("Either eigenvalues_pred or x and edge_index must be given.")

        # Compute loss
        lpl = 0
        el = 0
        eigenvalues_pred_tmp = None
        # Get DiffPool additional loss terms
        if self.encoder_type == DIFF_POOL:
            eigenvalues_pred_tmp, lpl, el = self(x=x.float(), edge_index=edge_index, batch_index=batch_index)

        # Get predictions if not given
        if eigenvalues_pred is None:
            if self.encoder_type == DIFF_POOL:
                eigenvalues_pred = eigenvalues_pred_tmp
            else:
                eigenvalues_pred = self(x=x.float(), edge_index=edge_index, batch_index=batch_index)

        # Compute MSE loss
        loss = criterion(eigenvalues_pred, eigenvalues_true)

        # Add additional DiffPool terms if required
        if self.encoder_type == DIFF_POOL:
            del eigenvalues_pred_tmp
            torch.cuda.empty_cache()
            loss = loss + lpl + el
        loss = loss.item()

        # Compute metrics
        metric_values = {}
        for metric in metrics:
            if metric != LOSS:
                metric_fn = METRICS_DICT[metric]
                metric_values[metric] = metric_fn(preds=eigenvalues_pred, target=eigenvalues_true)
        metric_values[LOSS] = loss

        return metric_values


def train_step_nm_net(model: EigenValueNMNet,
                      train_data: DataLoader,
                      optimizer,
                      device: torch.device,
                      criterion: Callable = MSELoss(),
                      logger: Optional[Logger] = None):
    # Put the model in training mode
    model.train()

    # Running average loss over the batches
    running_loss = 0.0
    steps: int = 1

    for data in iter(train_data):
        # Reset the optimizer gradients
        optimizer.zero_grad()
        data[NM_EIGENVALUES] = einops.rearrange(data[NM_EIGENVALUES], "(b f) -> b f", f=model.n_eigenvalues)
        loss = model.loss(eigenvalues_true=data[NM_EIGENVALUES].float().to(device),
                          x=data.x.float().to(device),
                          edge_index=data.edge_index.to(device),
                          batch_index=data.batch.to(device),
                          criterion=criterion)

        # Empty cuda cache
        torch.cuda.empty_cache()

        # Gradient computing
        loss.backward()

        # Advance the optimizer state performing gradient update
        optimizer.step()

        # Update running average loss
        running_loss = running_loss + 1 / steps * (loss.item() - running_loss)

        if logger is None:
            print(f"Steps: {steps}/{len(train_data)}, running loss {running_loss}")
        else:
            logger.log(f"Steps: {steps}/{len(train_data)}, running loss {running_loss}")
        steps += 1

        del loss
        del data
        torch.cuda.empty_cache()

    return float(running_loss)


@torch.no_grad()
def test_step_nm_net(model: EigenValueNMNet,
                     val_data: DataLoader,
                     device: torch.device,
                     criterion: Callable = MSELoss(),
                     metrics=frozenset([MAE, MAPE, PEARSON, CONCORDANCE])) -> dict[str, float]:
    # TODO: test this
    # put the model in evaluation mode
    model.eval()

    # Running average for loss, precision and AUCc = 0
    running_metrics = {metric: 0 for metric in metrics}
    steps: int = 1

    for data in iter(val_data):
        # move batch to device
        # data = data.to(device)
        data[NM_EIGENVALUES] = einops.rearrange(data[NM_EIGENVALUES], "(b f) -> b f", f=model.n_eigenvalues)
        metric_values = model.test(eigenvalues_true=data[NM_EIGENVALUES].float().to(device),
                                   x=data.x.float().to(device),
                                   edge_index=data.edge_index.to(device),
                                   batch_index=data.batch.to(device),
                                   criterion=criterion,
                                   metrics=metrics)
        torch.cuda.empty_cache()

        # Update metrics running average
        for metric in running_metrics:
            value = metric_values[metric]
            running_metrics[metric] = running_metrics[metric] + 1 / steps * (value - running_metrics[metric])
        steps += 1

    return running_metrics


def train_nm_net(model: EigenValueNMNet,
                 train_data: DataLoader,
                 val_data: DataLoader,
                 epochs: int,
                 optimizer,
                 experiment_path: str,
                 experiment_name: str,
                 early_stopping_patience: int = EARLY_STOP_PATIENCE,
                 early_stopping_delta: float = 0,
                 logger: Optional[Logger] = None,
                 criterion: Callable = MSELoss(),
                 metrics=frozenset([LOSS, RMSE, MAE, MAPE]),
                 monitored_metric: str = LOSS,
                 scheduler=None,
                 use_tensorboard_log: bool = False) -> (torch.nn.Module, dict):
    assert monitored_metric in metrics

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    experiment_path = os.path.join(experiment_path, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)  # create experiment directory if it doesn't exist

    # Instantiate the summary writer
    writer = None
    if use_tensorboard_log:
        writer = SummaryWriter(f'{experiment_path}_{experiment_name}_{epochs}_epochs')

    # Early-stopping monitor
    checkpoint_path = os.path.join(f"{experiment_path}", "checkpoint.pt")
    monitor = EarlyStopping(
        patience=early_stopping_patience,
        verbose=True,
        delta=early_stopping_delta,
        path=checkpoint_path,
        trace_func=logger.log
    )

    # Metric history trace object
    mht = MetricsHistoryTracer(
        metrics=[
            "train_loss",
            *[metric for metric in metrics]
        ],
        name="Classifier training metrics"
    )

    # Do validation step
    epoch_metrics = test_step_nm_net(
        model=model,
        val_data=val_data,
        device=device,
        criterion=criterion,
        metrics=metrics
    )

    torch.cuda.empty_cache()

    log_string = 'Epoch: {:d},'.format(0)
    for metric in epoch_metrics:
        metric_value = epoch_metrics[metric]
        if isinstance(metric_value, torch.Tensor):
            metric_value = metric_value.detach().cpu()
        log_string += f" {metric} {metric_value},"
    if logger is None:
        print(log_string)
    else:
        logger.log(log_string)

    # Tensorboard state update
    if use_tensorboard_log:
        for metric in epoch_metrics:
            metric_value = epoch_metrics[metric]
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.detach().cpu()
            writer.add_scalar(metric, metric_value, 0)

    # Check for early-stopping stuff
    monitor(epoch_metrics[monitored_metric], model)

    for epoch in range(0, epochs):
        # Do train step
        train_loss = train_step_nm_net(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            logger=None  # do not log epoch statistics to file
        )

        if scheduler is not None:
            scheduler.step()

        # Do validation step
        epoch_metrics = test_step_nm_net(
            model=model,
            val_data=val_data,
            device=device,
            criterion=criterion,
            metrics=metrics
        )

        torch.cuda.empty_cache()

        log_string = 'Epoch: {:d}, Train loss: {:.4f},'.format(epoch + 1, train_loss)
        for metric in epoch_metrics:
            metric_value = epoch_metrics[metric]
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.detach().cpu()
            log_string += f" {metric} {metric_value},"
        if logger is None:
            print(log_string)
        else:
            logger.log(log_string)

        # Tensorboard state update
        if use_tensorboard_log:
            writer.add_scalar('train_loss', train_loss, epoch)
            for metric in epoch_metrics:
                metric_value = epoch_metrics[metric]
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.detach().cpu()
                writer.add_scalar(metric, metric_value, epoch)

        # Check for early-stopping stuff
        monitor(epoch_metrics[monitored_metric], model)
        if monitor.early_stop:
            if logger is None:
                print(f"Epoch {epoch}: early stopping, restoring model checkpoint {checkpoint_path}...")
            else:
                logger.log(f"Epoch {epoch}: early stopping, restoring model checkpoint {checkpoint_path}...")
            break

        # Metrics history update
        mht.add_scalar('train_loss', train_loss)
        for metric in epoch_metrics:
            metric_value = epoch_metrics[metric]
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.detach().cpu()
            mht.add_scalar(metric, metric_value)

    # Plot the metrics
    mht.plot_metrics(
        [
            'train_loss',
            LOSS,
        ],
        figsize=FIGURE_SIZE_DEFAULT,
        traced_min_metric=LOSS,
        store_path=os.path.join(f"{experiment_path}", "loss.svg")
    )

    other_metrics = []
    for metric in metrics:
        if metric != LOSS:
            other_metrics.append(metric)
    mht.plot_metrics(
        other_metrics,
        figsize=FIGURE_SIZE_DEFAULT,
        store_path=os.path.join(f"{experiment_path}", "metrics.svg")
    )

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))

    # Do validation step
    final_metrics = test_step_nm_net(
        model=model,
        val_data=val_data,
        device=device,
        criterion=criterion,
        metrics=metrics
    )
    return model, final_metrics
