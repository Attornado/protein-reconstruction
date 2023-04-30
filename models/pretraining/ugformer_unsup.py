from typing import Optional, final, Union
import os
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, log_loss
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.dense import Linear
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from log.logger import Logger
from models.layers import SerializableModule
from models.pretraining.sampled_softmax import SampledSoftmax
from training.training_tools import EARLY_STOP_PATIENCE, EarlyStopping, MetricsHistoryTracer, FIGURE_SIZE_DEFAULT


DROPOUT_TRANSFORMER_BLOCK: final = 0.5
SOLVER_LOGISTIC_REGRESSION: final = "liblinear"
VOCAB_SIZE_KEY: final = "vocab_size"


class UGformerV1(SerializableModule):

    def __init__(self,
                 vocab_size: int,
                 feature_dim_size: int,
                 ff_hidden_size: int,
                 sampled_num: int,
                 num_self_att_layers: int,
                 num_gnn_layers: int,
                 embed_dim: Optional[int] = None,
                 n_heads: int = 1,
                 dropout: float = 0.5,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"
                 ):
        super(UGformerV1, self).__init__()

        # Init features
        self.__feature_dim_size: int = feature_dim_size
        self.__ff_hidden_size: int = ff_hidden_size
        self.__num_self_att_layers: int = num_self_att_layers  # a layer consists of a number of self-attention layers
        self.__num_gnn_layers: int = num_gnn_layers
        self.__vocab_size: int = vocab_size
        self.__sampled_num: int = sampled_num
        self.__device: torch.device = torch.device(device)
        self.__embed_dim: Optional[int] = embed_dim
        self.__n_heads: int = n_heads
        self.__dropout: float = dropout

        # Init projection layers if required
        self._projection: Linear = None if embed_dim is None else Linear(feature_dim_size, embed_dim)

        # Init transformer layers
        self._ugformer_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        for _ in range(self.num_gnn_layers):

            encoder_layers = TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=n_heads,
                dim_feedforward=self.ff_hidden_size,
                dropout=DROPOUT_TRANSFORMER_BLOCK
            )  # embed_dim must be divisible by num_heads
            self._ugformer_layers.append(TransformerEncoder(encoder_layers, self.num_self_att_layers))

        # Init dropout and sample-softmax layers
        self._dropout_layer = torch.nn.Dropout(dropout)
        self._ss = SampledSoftmax(self.vocab_size, self.sampled_num, self.embed_dim * self.num_gnn_layers, self.device)

    @property
    def feature_dim_size(self) -> int:
        return self.__feature_dim_size

    @property
    def ff_hidden_size(self) -> int:
        return self.__ff_hidden_size

    @property
    def num_self_att_layers(self) -> int:
        return self.__num_self_att_layers

    @property
    def num_gnn_layers(self) -> int:
        return self.__num_gnn_layers

    @property
    def vocab_size(self) -> int:
        return self.__vocab_size

    @property
    def embed_dim(self) -> int:
        return self.__embed_dim if self.__embed_dim is not None else self.__feature_dim_size

    @property
    def device(self) -> torch.device:
        return self.__device

    @property
    def n_heads(self) -> int:
        return self.__n_heads

    @property
    def sampled_num(self) -> int:
        return self.__sampled_num

    @property
    def dropout(self) -> float:
        return self.__dropout

    @property
    def learned_embeddings(self) -> torch.Tensor:
        return self._ss.weight

    # TODO: update to use indices from the entire train set
    def forward(self,
                x: torch.Tensor,
                sampled_neighbour_indices: torch.Tensor,
                input_y: Optional[torch.Tensor] = None
                ) -> (torch.Tensor, torch.Tensor):
        output_vectors = []  # should test output_vectors = [X_concat]
        input_tr = F.embedding(sampled_neighbour_indices, x)

        # Do a projection if required
        input_tr = input_tr if self._projection is None else self._projection(input_tr)

        for layer_idx in range(self.num_gnn_layers):

            # Index 0 is to take just the node of interest
            output_tr = self._ugformer_layers[layer_idx](input_tr)[0]

            # New input for next layer
            input_tr = F.embedding(sampled_neighbour_indices, output_tr)
            output_vectors.append(output_tr)

        # Concat output vectors to get the final node embeddings
        output_vectors = torch.cat(output_vectors, dim=1)
        output_vectors = self._dropout_layer(output_vectors)

        # Get sample-softmax logit loss
        logits = None
        if input_y is not None:
            logits = self._ss(output_vectors, input_y)

        # Return both loss and output embeddings
        return logits, output_vectors

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        return {
            "feature_dim_size": self.__feature_dim_size,
            "vocab_size": self.__vocab_size,
            "ff_hidden_size": self.__ff_hidden_size,
            "sampled_num": self.__sampled_num,
            "num_self_att_layers": self.__num_self_att_layers,
            "num_gnn_layers": self.__num_gnn_layers,
            "embed_dim": self.__embed_dim,
            "n_heads": self.__n_heads,
            "dropout": self.__dropout,
            "device": self.__device
        }


def get_global_node_indexes(batch_data: Data,
                            global_graph_indexes: dict[Union[str, int], tuple[int, int]]
                            ) -> torch.Tensor:

    # For all the start-end indexes in the node batch index (e.g  [0, 100, 250, 300] means graph 0 nodes are those from
    # 0 to 100, graph 1 nodes are those from 100 to 250, while graph 2 nodes are from 250 to 300)
    global_node_indexes = batch_data.batch.clone()
    for i in range(0, len(batch_data.ptr) - 1):
        # Get start and end indexes of the nodes of the i-th graph in the batch
        start_index = batch_data.ptr[i]
        end_index = batch_data.ptr[i + 1]

        # Get the global node start index of the i-th graph in the batch (e.g. 150, 200 means that, globally, the i-th
        # graph nodes start at the index 150 and end at 200)
        name = batch_data.name[i] if type(batch_data.name[i]) != list else batch_data.name[i][0]
        global_start_index = global_graph_indexes[name][0]
        global_end_index = global_graph_indexes[name][1]

        # Get the tensor of the global indexes of the nodes of the i-th graph in the batch, and put them in the tensor
        global_node_indexes[start_index:end_index] = torch.arange(start=global_start_index, end=global_end_index)

    return global_node_indexes


def compute_global_graph_indexes(train_data: DataLoader,
                                 path: Optional[str] = None,
                                 use_tqdm: bool = False
                                 ) -> dict[Union[str, int], Union[int, tuple[int, int]]]:
    global_node_indexes = {}
    iterable = tqdm(enumerate(train_data), desc="Computing graph global indexes") if use_tqdm else enumerate(train_data)
    node_count = 0
    last_graph_name = None

    for i, data in iterable:
        # Get the graph name, assuming batch size 1
        name = data.name[0] if type(data.name[0]) != list else data.name[0][0]

        # Get the global node index range for the current graph, and increment the global node count
        start_index = node_count
        end_index = node_count + data.x.shape[0]
        node_count += data.x.shape[0]

        # Store index range in dictionary
        global_node_indexes[name] = (start_index, end_index)

        # Trace last graph name to get the final vocab size (total node number)
        last_graph_name = name

    # Get the final vocab size (total node number) and store it in dictionary
    vocab_size = global_node_indexes[last_graph_name][1]
    global_node_indexes[VOCAB_SIZE_KEY] = vocab_size

    # Store the dictionary to given path if required
    if path is not None:
        torch.save(global_node_indexes, path)

    return global_node_indexes


def get_batch_data(batch_data: Data,
                   n_neighbours: int,
                   device: torch.device,
                   global_graph_indexes: Optional[dict[Union[str, int], tuple[int, int]]] = None
                   ) -> (torch.Tensor, torch.Tensor):

    # Get the adjacency list of nodes, for each node
    node_adj_dict = {}
    for i in range(0, len(batch_data.edge_index[0])):
        u = batch_data.edge_index[0][i]
        v = batch_data.edge_index[1][i]
        if u not in node_adj_dict:
            node_adj_dict[u]: list[int] = []
        node_adj_dict[u].append(v)

    # Sample neighbours
    input_neighbors = []
    for node in range(batch_data.x.shape[0]):
        if node in node_adj_dict:
            # Sample the neighbours using numpy, always putting the current node first
            input_neighbors.append(
                [node] + list(np.random.choice(node_adj_dict[node], n_neighbours, replace=True)))
        else:
            input_neighbors.append([node for _ in range(n_neighbours + 1)])

    selected_neighbours_indices = np.array(input_neighbors)
    selected_neighbours_indices = torch.transpose(
        torch.from_numpy(selected_neighbours_indices),
        dim0=0,
        dim1=1
    ).to(device)  # [seq_length, batch_size] for pytorch transformer, not [batch_size, seq_length]

    if global_graph_indexes is not None:
        # Get the global node indexes
        node_indexes_y = get_global_node_indexes(batch_data=batch_data,
                                                 global_graph_indexes=global_graph_indexes).to(device)
    else:
        # We need just all the nodes because we already work on a single batch
        node_indexes_y = torch.arange(0, batch_data.x.shape[0]).to(device)

    return selected_neighbours_indices, node_indexes_y


def train_step_ugformer_unsup(model: UGformerV1,
                              train_data: DataLoader,
                              optimizer,
                              n_neighbours: int,
                              device: torch.device,
                              global_graph_indexes: dict[Union[str, int], tuple[int, int]],
                              logger: Optional[Logger] = None):
    # Put the model in training mode
    model.train()

    # Running average loss over the batches
    running_loss = 0.0
    steps: int = 1

    for data in iter(train_data):
        selected_neighbours_indices, node_indices_y = get_batch_data(batch_data=data,
                                                                     n_neighbours=n_neighbours,
                                                                     device=device,
                                                                     global_graph_indexes=global_graph_indexes)
        # Reset the optimizer gradients
        optimizer.zero_grad()

        # Get sample-softmax loss
        loss, _ = model(x=data.x.float().to(device),
                        sampled_neighbour_indices=selected_neighbours_indices,
                        input_y=node_indices_y)
        loss = loss.mean()  # For some reason, loss is not averaged by default

        del _
        torch.cuda.empty_cache()

        # Gradient update
        loss.mean().backward()

        # Advance the optimizer state
        optimizer.step()

        # Update running average loss
        running_loss = running_loss + 1 / steps * (loss.item() - running_loss)

        if logger is None:
            print(f"Steps: {steps}/{len(train_data)}, running loss {running_loss}")
        else:
            logger.log(f"Steps: {steps}/{len(train_data)}, running loss {running_loss}")

        del loss
        del data
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        steps += 1

    return float(running_loss)


def _divide_and_group_by_graph(output_vectors: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    grouped_output_vectors = []
    # For each graph in the batch
    for i in range(1, len(ptr)):
        # Get output vectors for the corresponding graph
        graph_x = torch.sum(output_vectors[ptr[i - 1]:ptr[i]], dim=0)
        grouped_output_vectors.append(graph_x)
    return torch.stack(grouped_output_vectors)


@torch.no_grad()
def test_step_ugformer_unsup(model: UGformerV1, train_data: DataLoader, val_data: DataLoader, n_neighbours: int,
                             device: torch.device):
    # TODO: update to use indices from the entire train set

    # Put the model in evaluation mode
    model.eval()

    # Running average for loss, precision and AUC
    steps: int = 1
    x_values_train = []
    y_values_train = []
    for data in iter(train_data):
        # Get the sampled neighbours
        selected_neighbours_indices, _ = get_batch_data(batch_data=data, n_neighbours=n_neighbours, device=device,
                                                        global_graph_indexes=None)

        # Get output embeddings
        _, output_vectors = model(x=data.x.float().to(device), sampled_neighbour_indices=selected_neighbours_indices,
                                  input_y=None)

        output_vectors = _divide_and_group_by_graph(output_vectors, ptr=data.ptr)
        x_values_train.append(output_vectors.detach().cpu())
        y_values_train.append(data.y.detach().cpu())
        del data
        del selected_neighbours_indices
        del _, output_vectors
        torch.cuda.empty_cache()
        steps += 1

    x_values_val = []
    y_values_val = []
    steps: int = 1
    for data in iter(val_data):
        selected_neighbours_indices, node_indices_y = get_batch_data(batch_data=data, n_neighbours=n_neighbours,
                                                                     device=device)
        # Move batch to device
        # data = data.to(device)

        # Get sample-softmax loss
        _, output_vectors = model(x=data.x.float().to(device),
                                  sampled_neighbour_indices=selected_neighbours_indices,
                                  input_y=node_indices_y)

        output_vectors = _divide_and_group_by_graph(output_vectors, ptr=data.ptr)
        x_values_val.append(output_vectors.detach().cpu())
        y_values_val.append(data.y.detach().cpu())
        del data
        del selected_neighbours_indices
        del _, output_vectors
        torch.cuda.empty_cache()
        steps += 1

    x_values_val = torch.cat(x_values_val, dim=0).detach().cpu().numpy()
    y_values_val = torch.cat(y_values_val, dim=0).detach().cpu().numpy()
    x_values_train = torch.cat(x_values_train, dim=0).detach().cpu().numpy()
    y_values_train = torch.cat(y_values_train, dim=0).detach().cpu().numpy()
    cls = LogisticRegression(solver=SOLVER_LOGISTIC_REGRESSION, tol=0.001, class_weight='balanced')

    cls.fit(x_values_train, y_values_train)
    acc = cls.score(x_values_val, y_values_val)
    probs = cls.predict_proba(x_values_val)
    # Use OvO because it's insensitive to class imbalance
    auc = roc_auc_score(y_true=y_values_val, y_score=probs, average="macro", multi_class="ovo")

    return acc, auc


def train_ugformer_unsup_inductive(model: UGformerV1,
                                   train_data: DataLoader,
                                   val_data: DataLoader,
                                   epochs: int,
                                   optimizer,
                                   experiment_path: str,
                                   experiment_name: str,
                                   n_neighbours: int,
                                   val_train_data: Optional[DataLoader] = None,
                                   global_graph_indexes: Optional[dict[Union[str, int], tuple[int, int]]] = None,
                                   early_stopping_patience: int = EARLY_STOP_PATIENCE,
                                   early_stopping_delta: float = 0,
                                   logger: Optional[Logger] = None,
                                   use_tensorboard_log: bool = False
                                   ) -> (torch.nn.Module, dict):
    # TODO: test this
    # Move model to device
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    experiment_path = os.path.join(experiment_path, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)  # create experiment directory if it doesn't exist

    # Instantiate the summary writer
    if use_tensorboard_log:
        writer = SummaryWriter(f'{experiment_path}_{experiment_name}_{epochs}_epochs')
    else:
        writer = None

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
            'train_loss',
            'AUC',
            'accuracy'
        ],
        name="Classifier training metrics"
    )

    for epoch in range(0, epochs):
        # Do train step
        train_loss = train_step_ugformer_unsup(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            n_neighbours=n_neighbours,
            device=device,
            global_graph_indexes=global_graph_indexes,
            logger=None  # do not log epoch statistics to file
        )

        # Do validation step
        if val_train_data is None:
            val_train_data = train_data
        acc, auc = test_step_ugformer_unsup(
            model=model,
            train_data=val_train_data,
            val_data=val_data,
            n_neighbours=n_neighbours,
            device=device
        )

        if logger is None:
            print(
                'Epoch: {:d}, Train loss: {:.4f}, Accuracy: {:.4f}, AUC: {:.4f}'
                .format(epoch + 1, train_loss, acc, auc)
            )
        else:
            logger.log(
                'Epoch: {:d}, Train loss: {:.4f}, Accuracy: {:.4f}, AUC: {:.4f}'
                .format(epoch + 1, train_loss, acc, auc)
            )

        # Tensorboard state update
        if use_tensorboard_log:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('accuracy', acc, epoch)
            writer.add_scalar('auc', auc, epoch)

        # Check for early-stopping stuff
        monitor(-auc, model)  # check if accuracy/auc is better
        if monitor.early_stop:
            if logger is None:
                print(f"Epoch {epoch}: early stopping, restoring model checkpoint {checkpoint_path}...")
            else:
                logger.log(f"Epoch {epoch}: early stopping, restoring model checkpoint {checkpoint_path}...")
            break

        # Metrics history update
        mht.add_scalar('train_loss', train_loss)
        mht.add_scalar('AUC', auc)
        mht.add_scalar('accuracy', acc)

    # Plot the metrics
    mht.plot_metrics(
        [
            'train_loss',
            'AUC',
        ],
        figsize=FIGURE_SIZE_DEFAULT,
        traced_max_metric='AUC',
        store_path=os.path.join(f"{experiment_path}", "loss.svg")
    )

    mht.plot_metrics(
        [
            "AUC",
        ],
        figsize=FIGURE_SIZE_DEFAULT,
        traced_max_metric='AUC',
        store_path=os.path.join(f"{experiment_path}", "auc.svg")
    )

    mht.plot_metrics(
        [
            'accuracy',
        ],
        figsize=FIGURE_SIZE_DEFAULT,
        traced_max_metric='accuracy',
        store_path=os.path.join(f"{experiment_path}", "accuracy.svg")
    )

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))

    acc, auc = test_step_ugformer_unsup(
        model=model,
        train_data=train_data if val_train_data is None else val_train_data,
        val_data=val_data,
        n_neighbours=n_neighbours,
        device=device
    )

    metrics = {
        "accuracy": acc,
        "auc": auc
    }
    return model, metrics
