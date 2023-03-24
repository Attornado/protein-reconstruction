import abc
import os
from abc import abstractmethod
from typing import Callable, Optional, Union, Iterable
from log.logger import Logger
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torchmetrics.functional import accuracy, precision, recall, f1_score
from models.layers import SerializableModule
from training.training_tools import EarlyStopping, MetricsHistoryTracer, FIGURE_SIZE_DEFAULT, EARLY_STOP_PATIENCE


class ClassificationLoss(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        self._loss: Optional[Callable] = None

    def forward(self, targets: torch.Tensor, *outputs: torch.Tensor) -> torch.Tensor:
        """
        :param targets: labels
        :param outputs: predictions
        :return: loss value
        """
        outputs = outputs[0]
        loss = self._loss(outputs, targets)
        return loss

    def get_accuracy(self, targets: torch.Tensor, *outputs: torch.Tensor) -> float:
        outputs: torch.Tensor = outputs[0]
        acc = self._calculate_accuracy(outputs, targets)
        return acc

    @abstractmethod
    def _get_correct(self, outputs):
        raise NotImplementedError()

    def _calculate_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        correct = self._get_correct(outputs)
        return float(100. * (correct == targets).sum().float() / targets.size(0))


class MulticlassClassificationLoss(ClassificationLoss):
    def __init__(self, weights: Optional[Union[torch.Tensor, Iterable]] = None, reduction: Optional[str] = None,
                 label_smoothing: float = 0.0):
        super().__init__()

        if weights is None or isinstance(weights, torch.Tensor):
            self.__weights: Optional[torch.Tensor] = weights
        else:
            self.__weights: Optional[torch.Tensor] = torch.tensor(weights)

        if reduction is not None:
            self._loss: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss(reduction=reduction, weight=weights,
                                                                              label_smoothing=label_smoothing)
        else:
            self._loss: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss(weight=weights,
                                                                              label_smoothing=label_smoothing)

    @property
    def weights(self) -> Optional[torch.Tensor]:
        return self.__weights

    @weights.setter
    def weights(self, weights: Optional[Union[torch.Tensor, Iterable]]):
        if weights is None or isinstance(weights, torch.Tensor):
            self.__weights: Optional[torch.Tensor] = weights
        else:
            self.__weights: Optional[torch.Tensor] = torch.tensor(weights)

    def _get_correct(self, outputs):
        return torch.argmax(outputs, dim=1)


class GraphClassifier(SerializableModule):
    def __init__(self, dim_features: int, dim_target: int, config: dict):
        super(GraphClassifier, self).__init__()
        self.__in_channels: int = dim_features
        self.__dim_target: int = dim_target
        self.__config_dict: dict = config

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    @in_channels.setter
    def in_channels(self, in_channels: int):
        self.__in_channels = in_channels

    @property
    def dim_target(self) -> int:
        return self.__dim_target

    @dim_target.setter
    def dim_target(self, dim_target: int):
        self.__dim_target = dim_target

    @property
    def config_dict(self) -> dict:
        return self.__config_dict

    @config_dict.setter
    def config_dict(self, config_dict: dict):
        self.__config_dict = config_dict

    def test(self, x: torch.Tensor, edge_index: torch.Tensor, y, batch_index: torch.Tensor = None,
             criterion: ClassificationLoss = MulticlassClassificationLoss(), top_k: Optional[int] = None,
             *args, **kwargs) -> (float, Optional[float], float, float, float, float):
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

        # Get the number of classes
        n_classes = self.dim_target

        # Get predictions
        y_hat = self(x.float(), edge_index, batch_index, *args, **kwargs)

        # Compute loss
        loss = self.loss(y_hat=y_hat, y=y, criterion=criterion)

        # Compute the metrics
        acc = accuracy(preds=y_hat, target=y, task='multiclass', num_classes=n_classes, average="macro")
        if top_k is not None:
            top_k_acc = float(accuracy(preds=y_hat, target=y, task='multiclass', num_classes=n_classes, top_k=top_k,
                                       average="macro"))
        else:
            top_k_acc = None
        prec = precision(preds=y_hat, target=y, task='multiclass', num_classes=n_classes, average="macro")
        rec = recall(preds=y_hat, target=y, task='multiclass', num_classes=n_classes, average="macro")
        f1 = f1_score(preds=y_hat, target=y, task='multiclass', num_classes=n_classes, average="macro")

        return float(loss), float(acc), top_k_acc, prec, rec, f1

    def loss(self, y, x: Optional[torch.Tensor] = None, edge_index: Optional[torch.Tensor] = None,
             batch_index: Optional[torch.Tensor] = None, y_hat: Optional[torch.Tensor] = None,
             criterion: ClassificationLoss = MulticlassClassificationLoss(),
             additional_terms: list[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:

        # If predictions are not given, compute them using the model
        if y_hat is None:
            y_hat = self(x.float(), edge_index, batch_index, *args, **kwargs)

        # Compute loss with given criterion
        loss = criterion(y, y_hat)

        # Add pre-computed additional loss terms to the loss
        if additional_terms is not None:
            for additional_term in additional_terms:
                loss = loss + additional_term

        return loss

    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        raise NotImplementedError(f"Each {self.__class__} class has to implement the forward() method")

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        return {
            "dim_features": self.in_channels,
            "dim_target": self.dim_target,
            "config": self.config_dict
        }


def train_step_classifier(model: GraphClassifier, train_data: DataLoader, optimizer, device: torch.device,
                          criterion: ClassificationLoss = MulticlassClassificationLoss(),
                          logger: Optional[Logger] = None):
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
        y_hat = model(data.x.float(), data.edge_index, data.batch)

        loss = model.loss(y=data.y, y_hat=y_hat, criterion=criterion, additional_terms=None)

        # Gradient update
        loss.backward()
        # Advance the optimizer state
        optimizer.step()

        # Update running average loss
        running_loss = running_loss + 1 / steps * (loss.item() - running_loss)

        if logger is None:
            print(f"Steps: {steps}/{len(train_data)}, running loss {running_loss}")
        else:
            logger.log(f"Steps: {steps}/{len(train_data)}, running loss {running_loss}")
        steps += 1

    return float(running_loss)


@torch.no_grad()
def test_step_classifier(model: GraphClassifier, val_data: DataLoader, device: torch.device,
                         use_edge_weight: bool = False, use_edge_attr: bool = False, top_k: int = 3,
                         criterion: ClassificationLoss = MulticlassClassificationLoss()):
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


def train_classifier(model: GraphClassifier, train_data: DataLoader, val_data: DataLoader, epochs: int, optimizer,
                     experiment_path: str, experiment_name: str, early_stopping_patience: int = EARLY_STOP_PATIENCE,
                     early_stopping_delta: float = 0, top_k: int = 3, logger: Optional[Logger] = None,
                     criterion: ClassificationLoss = MulticlassClassificationLoss(),
                     use_tensorboard_log: bool = False) -> (torch.nn.Module, dict):
    # TODO: test this
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    experiment_path = os.path.join(experiment_path, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)  # create experiment directory if it doesn't exist

    # Instantiate the summary writer
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
            criterion=criterion,
            logger=None  # do not log epoch statistics to file
        )

        # Do validation step
        avg_precision, avg_recall, avg_accuracy, avg_topk_accuracy, avg_f1, val_loss = test_step_classifier(
            model=model,
            val_data=val_data,
            device=device,
            top_k=top_k,
            criterion=criterion
        )

        if logger is None:
            print(
                'Epoch: {:d}, Train loss: {:.4f}, Validation loss {:.4f}, Average accuracy: {:.4f}, '
                'Average top-{:d} accuracy: {:.4f}, Average precision: {:.4f}, Average recall: {:.4f}, '
                'Average F1: {:.4f}, '
                .format(epoch + 1, train_loss, val_loss, avg_accuracy, top_k,
                        avg_topk_accuracy, avg_precision, avg_recall, avg_f1)
            )
        else:
            logger.log(
                'Epoch: {:d}, Train loss: {:.4f}, Validation loss {:.4f}, Average accuracy: {:.4f}, '
                'Average top-{:d} accuracy: {:.4f}, Average precision: {:.4f}, Average recall: {:.4f}, '
                'Average F1: {:.4f}, '
                .format(epoch + 1, train_loss, val_loss, avg_accuracy, top_k,
                        avg_topk_accuracy, avg_precision, avg_recall, avg_f1)
            )

        # Tensorboard state update
        if use_tensorboard_log:
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
            if logger is None:
                print(f"Epoch {epoch}: early stopping, restoring model checkpoint {checkpoint_path}...")
            else:
                logger.log(f"Epoch {epoch}: early stopping, restoring model checkpoint {checkpoint_path}...")
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

    avg_precision, avg_recall, avg_accuracy, avg_topk_accuracy, avg_f1, val_loss = test_step_classifier(
        model=model,
        val_data=val_data,
        device=device,
        top_k=top_k,
        criterion=criterion
    )
    metrics = {
        "precision": avg_precision,
        "recall": avg_recall,
        "accuracy": avg_accuracy,
        "avg_topk_accuracy": avg_topk_accuracy,
        "f1": avg_f1,
        "val_loss": val_loss
    }
    return model, metrics
