import os
from typing import Optional, Any, Type
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models.autoencoder import GAE
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import negative_sampling
from models.layers import SerializableModule
from training.training_tools import FIGURE_SIZE_DEFAULT, MetricsHistoryTracer, EarlyStopping, EARLY_STOP_PATIENCE


class GAEv2(GAE, SerializableModule):

    def __init__(self, encoder: SerializableModule, decoder: Optional[SerializableModule] = None):
        """
        GAE sub-class with a simple forward function implemented.

        :param encoder: The encoder network
        :type encoder: torch.nn.Module
        :param decoder: Optional[torch.nn.Module] = None
        :type decoder: Optional[torch.nn.Module]
        """
        super(GAEv2, self).__init__(encoder=encoder, decoder=decoder)
        self.__serialize_decoder = decoder is not None  # True if decoder is not None, False otherwise

    def forward(self, x, edge_index, sigmoid: bool = True, *args, **kwargs):
        z = self.encode(x, edge_index, *args, **kwargs)
        adj_rec = self.decode(z, sigmoid=sigmoid)
        return adj_rec

    # noinspection PyTypedDict
    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        constructor_params = {}
        constructor_params["encoder"] = {
            "state_dict": self.encoder.state_dict(),
            "constructor_params": self.encoder.serialize_constructor_params()
        }
        if self.__serialize_decoder:
            constructor_params["decoder"] = {
                "state_dict": self.decoder.state_dict(),
                "constructor_params": self.decoder.serialize_constructor_params()
            }
        else:
            constructor_params["decoder"] = None

        return constructor_params

    # noinspection PyMethodOverriding
    @classmethod
    def from_constructor_params(cls, constructor_params: dict, encoder_constructor: Type[SerializableModule],
                                decoder_constructor: Optional[Type[SerializableModule]] = None, *args, **kwargs):
        # Get encoder constructor params/state dict and construct it
        enc_state_dict = constructor_params["encoder"]["state_dict"]
        enc_constructor_params = constructor_params["encoder"]["constructor_params"]
        encoder = encoder_constructor.from_constructor_params(enc_constructor_params)  # construct encoder
        encoder.load_state_dict(state_dict=enc_state_dict)  # set weights

        # If required, get decoder constructor params/state dict and construct it
        if constructor_params["decoder"] is not None:
            dec_state_dict = constructor_params["decoder"]["state_dict"]
            dec_constructor_params = constructor_params["decoder"]["constructor_params"]
            decoder = decoder_constructor.from_constructor_params(dec_constructor_params)  # construct decoder
            decoder.load_state_dict(state_dict=dec_state_dict)  # set weights
        else:
            decoder = None

        return cls(encoder=encoder, decoder=decoder)


def train_step_gae(model: GAEv2, train_data: DataLoader, optimizer, device, use_edge_weight: bool = False,
                   use_edge_attr: bool = False):

    model.train()  # put the model in training mode

    running_loss = 0.0  # running average loss over the batches
    steps: int = 1

    for data in iter(train_data):
        data = data.to(device)  # move batch to device
        optimizer.zero_grad()  # reset the optimizer gradients

        # Encoder output
        if use_edge_weight and use_edge_attr:
            z = model.encode(data.x, data.edge_index, edge_attr=data.edge_attr, edge_weight=data.edge_weight)
        elif use_edge_attr:
            z = model.encode(data.x, data.edge_index, edge_attr=data.edge_attr)
        elif use_edge_weight:
            z = model.encode(data.x, data.edge_index, edge_weight=data.edge_weight)
        else:
            z = model.encode(data.x, data.edge_index)

        loss = model.recon_loss(z, data.edge_index)  # reconstruction

        loss.backward()  # gradient update
        optimizer.step()  # advance the optimizer state

        # Update running average loss
        running_loss = running_loss + 1 / steps * (loss.item() - running_loss)
        steps += 1

    return float(running_loss)


@torch.no_grad()
def test_step_gae(model: GAEv2, val_data: DataLoader, device, use_edge_weight: bool = False,
                  use_edge_attr: bool = False):
    model.eval()  # put the model in evaluation mode

    # Running average for loss, precision and AUC
    running_val_loss = 0.0
    running_auc = 0.0
    running_precision = 0.0
    steps: int = 1

    for data in iter(val_data):
        data = data.to(device)  # move batch to device

        # Encoder output
        if use_edge_weight and use_edge_attr:
            z = model.encode(data.x, data.edge_index, edge_attr=data.edge_attr, edge_weight=data.edge_weight)
        elif use_edge_attr:
            z = model.encode(data.x, data.edge_index, edge_attr=data.edge_attr)
        elif use_edge_weight:
            z = model.encode(data.x, data.edge_index, edge_weight=data.edge_weight)
        else:
            z = model.encode(data.x, data.edge_index)

        loss = model.recon_loss(z, data.edge_index)  # reconstruction loss
        running_val_loss = running_val_loss + 1/steps * (loss.item() - running_val_loss)  # update loss running average

        # Update AUC and precision running averages
        neg_edge_index = negative_sampling(data.edge_index, z.size(0))
        auc, avg_precision = model.test(z, pos_edge_index=data.edge_index, neg_edge_index=neg_edge_index)
        running_auc = running_auc + 1/steps * (auc - running_auc)
        running_precision = running_precision + 1/steps * (avg_precision - running_precision)
        steps += 1

    return float(running_val_loss), running_auc, running_precision


def train_vgae(model: GAEv2, train_data: DataLoader, val_data: DataLoader, epochs: int, optimizer,
               experiment_path: str, experiment_name: str, use_edge_weight: bool = False, use_edge_attr: bool = False,
               early_stopping_patience: int = EARLY_STOP_PATIENCE, early_stopping_delta: float = 0) -> torch.nn.Module:
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
        metrics=['train_loss', 'val_loss', 'auc_val', 'avg_precision_val'],
        name="VGAE training metrics"
    )

    for epoch in range(0, epochs):
        # Do train step
        train_loss = train_step_gae(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            device=device,
            use_edge_weight=use_edge_weight,
            use_edge_attr=use_edge_attr
        )

        # Do validation step
        val_loss, auc, avg_precision = test_step_gae(
            model=model,
            val_data=val_data,
            device=device,
            use_edge_weight=use_edge_weight,
            use_edge_attr=use_edge_attr
        )

        print('Epoch: {:d}, Train loss: {:.4f}, Validation loss {:.4f}, '
              'AUC: {:.4f}, Average precision: {:.4f}'.format(epoch, train_loss, val_loss, auc, avg_precision))

        # Tensorboard state update
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('auc_val', auc, epoch)  # new line
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('avg_precision_val', avg_precision, epoch)  # new line

        # Check for early-stopping stuff
        monitor(val_loss, model)
        if monitor.early_stop:
            print(f"Epoch {epoch}: early stopping, restoring model checkpoint {checkpoint_path}...")
            break

        # Metrics history update
        mht.add_scalar('train_loss', train_loss)
        mht.add_scalar('val_loss', val_loss)
        mht.add_scalar('auc_val', auc)
        mht.add_scalar('avg_precision_val', avg_precision)

    # Plot the metrics
    mht.plot_metrics(
        ['train_loss', 'val_loss'],
        figsize=FIGURE_SIZE_DEFAULT,
        traced_min_metric='val_loss',
        store_path=os.path.join(f"{experiment_path}", "loss.svg")
    )

    mht.plot_metrics(
        ['auc_val'],
        figsize=FIGURE_SIZE_DEFAULT,
        traced_min_metric='auc_val',
        store_path=os.path.join(f"{experiment_path}", "auc.svg")
    )

    mht.plot_metrics(
        ['avg_precision_val'],
        figsize=FIGURE_SIZE_DEFAULT,
        traced_min_metric='avg_precision_val',
        store_path=os.path.join(f"{experiment_path}", "avg_precision.svg")
    )

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))

    return model
