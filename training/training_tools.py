import os
import random
from typing import final, Iterable, Optional, Literal, Callable
import numpy as np
import torch
import matplotlib.pyplot as plt


ACCURACY_METRIC: final = "acc"
VAL_LOSS_METRIC: final = "val_loss"
TRAIN_LOSS_METRIC: final = "train_loss"
F1_METRIC: final = "f1"
DEFAULT_METRICS: final = frozenset([TRAIN_LOSS_METRIC, VAL_LOSS_METRIC])
METRIC_TRACER_DEFAULT: final = "metric_tracer"
FIGURE_SIZE_DEFAULT: final = (10, 8)
EARLY_STOP_PATIENCE: final = 10


# Class from Bjarten's early-stopping-pytorch repository. All credits go to him and the other contributors.
# Please check the original source on the repository https://github.com/Bjarten/early-stopping-pytorch.git
class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self,
                 patience: int = EARLY_STOP_PATIENCE,
                 verbose: bool = False,
                 delta: float = 0,
                 path: str = 'checkpoint.pt',
                 trace_func: Callable = print,
                 monitored_metric_name: str = VAL_LOSS_METRIC):
        """
        Args:
            patience (int): How long to wait after last time validation loss/metric improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss/metric  improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
            monitored_metric_name (str): name of the monitored metric.
                            Default: VAL_LOSS_METRIC
        """
        self.__patience: int = patience
        self.__verbose: bool = verbose
        self.__counter: int = 0
        self.__best_score = None
        self.early_stop = False
        self.monitored_metric_best = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.monitored_metric_name = monitored_metric_name

    @property
    def patience(self) -> int:
        return self.__patience

    @patience.setter
    def patience(self, patience: int):
        self.__patience = patience

    @property
    def verbose(self) -> bool:
        return self.__verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        self.__verbose = verbose

    @property
    def counter(self) -> int:
        return self.__counter

    def __call__(self, monitored_metric, model):

        score = -monitored_metric

        if self.__best_score is None:
            self.__best_score = score
            self.save_checkpoint(monitored_metric, model)
        elif score < self.__best_score + self.delta:
            self.__counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.__best_score = score
            self.save_checkpoint(monitored_metric, model)
            self.__counter = 0

    def save_checkpoint(self, monitored_metric, model):
        """Saves model when monitored metric improves."""
        if self.verbose:
            self.trace_func(
                f'{self.monitored_metric_name} improved ({self.monitored_metric_best:.6f} --> '
                f'{monitored_metric:.6f}).  Saving model ...'
            )
        torch.save(model.state_dict(), self.path)
        self.monitored_metric_best = monitored_metric


class MetricsHistoryTracer(object):
    """
    Simple class to trace arbitrary metrics during training.

    :param metrics: an iterable of metrics to trace during training, defaults to DEFAULT_METRICS constant.
    :type metrics: Iterable[str]
    :param name: metric t
    """

    def __init__(self, metrics: Iterable[str] = DEFAULT_METRICS, name: str = METRIC_TRACER_DEFAULT):
        self.__metrics = {metric: np.array([], dtype=np.float64) for metric in metrics}  # initialize metric dictionary
        self.__name = name

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def metrics(self) -> list[str]:
        return list(self.__metrics.keys())

    def get_metric(self, metric: str):
        """
       Returns the value of the metric specified by the user.

        :param metric: The name of the metric to get
        :type metric: str
        :return: The values of the metric.
        """
        if metric not in self.__metrics:
            raise ValueError(f"Metric {metric} is not traced by this object.")
        return self.__metrics[metric]

    def add_scalar(self, metric: str, value):
        """
        Adds a scalar value to the metric specified by the user.

        :param metric: metric name
        :type metric: str
        :param value: The value to be added to the metric
        """
        if metric not in self.__metrics:
            raise ValueError(f"Metric {metric} is not traced by this object.")
        else:
            self.__metrics[metric] = np.concatenate((self.__metrics[metric], [value]), -1)

    def add_multiple(self, metric: str, values: np.ndarray):
        """
        Takes in a metric name and a numpy array of values, and adds the values to the metric

        :param metric: metric name
        :type metric: str
        :param values: metric values to add
        :type values: np.ndarray
        """
        if metric not in self.__metrics:
            raise ValueError(f"Metric {metric} is not traced by this object.")

        if values.ndim != 1:
            raise ValueError(f"Given metric arrays must be 1-dimensional, {values.ndim}-dimensional given.")
        else:
            self.__metrics[metric] = np.concatenate((self.__metrics[metric], values), -1)

    def plot_metrics(self, metrics: Optional[Iterable[str]] = None, figsize: tuple[int, int] = FIGURE_SIZE_DEFAULT,
                     traced_min_metric: Optional[str] = None, traced_max_metric: Optional[str] = None,
                     store_path: Optional[str] = None):

        if metrics is None:
            metrics = list(self.__metrics.keys())

        # Create the figure
        plt.style.use("dark_background")  # set dark background
        fig = plt.figure(figsize=figsize)
        title = f"{self.name}: "
        x_limit = -1
        y_limit = -1

        # Plot each given metric
        for metric in metrics:
            if metric not in self.__metrics:
                raise ValueError(f"Metric {metric} is not traced by this object.")

            metric_history: np.ndarray = self.__metrics[metric]

            if len(metric_history) > 0:
                plt.plot(range(1, len(metric_history) + 1), metric_history, label=f'{metric}')
                if metric == traced_min_metric:
                    # Find position of the lowest metric point
                    min_position = np.argmin(metric_history) + 1
                    plt.axvline(min_position, linestyle='--', color='r', label=f'{metric} minimum')

            if metric == traced_max_metric:
                # Find position of the highest metric point
                max_position = np.argmax(metric_history) + 1
                plt.axvline(max_position, linestyle='--', color='r', label=f'{metric} maximum')

            if len(metric_history) > x_limit:
                x_limit = len(metric_history)

            max_m = np.max(metric_history)
            if np.abs(max_m) > y_limit:
                y_limit = np.abs(max_m)

            title = f"{title} {metric}"

        if x_limit == -1:
            x_limit = 1

        if y_limit == -1:
            y_limit = 1

        # Other parameters
        plt.title(self.name)  # set the plot title
        plt.ylabel('metric')
        plt.ylim(0, y_limit + int(y_limit)/50)  # consistent scale
        plt.xlim(0, x_limit + int(x_limit)/50)  # consistent scale
        plt.xlabel('epochs')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if store_path is not None:
            fig.savefig(store_path, bbox_inches='tight')

        # Show plot
        plt.show()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.allow_tf32 = False
