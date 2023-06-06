import os
from typing import Optional, Any, Type
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.nn.inits import reset
from torch_geometric.nn.models.autoencoder import VGAE
from torch_geometric.utils import negative_sampling
from models.pretraining.decoders import GraphDecoder
from models.layers import SerializableModule
from training.training_tools import MetricsHistoryTracer, EarlyStopping, EARLY_STOP_PATIENCE, FIGURE_SIZE_DEFAULT


class VGEncoder(SerializableModule):
    def __init__(self, encoder_mu: SerializableModule, encoder_logstd: Optional[SerializableModule] = None,
                 shared_encoder: Optional[SerializableModule] = None, standalone: bool = False):
        super(VGEncoder, self).__init__()
        self._encoder_mu = encoder_mu

        self._shared_encoder = shared_encoder
        self.__shared_encoder_given = shared_encoder is not None
        self.__standalone = standalone

        # If no encoder_logstd is given, deep copy the encoder_mu
        self.__encoder_logstd_given = True
        if encoder_logstd is None:
            self.__encoder_logstd_given = False
            encoder_logstd = type(encoder_mu).from_constructor_params(encoder_mu.serialize_constructor_params())

        self._encoder_logstd = encoder_logstd

    @property
    def standalone(self) -> bool:
        return self.__standalone

    @standalone.setter
    def standalone(self, standalone: bool):
        self.__standalone = standalone

    def forward(self, x, edge_index, *args, **kwargs):

        # If a shared encoder is given, apply it before encoder the mu and log(std) ones
        if self._shared_encoder is not None:
            x = self._shared_encoder(x, edge_index, *args, **kwargs)

        if self.standalone:
            return self._encoder_mu(x, edge_index, *args, **kwargs)

        return self._encoder_mu(x, edge_index, *args, **kwargs), self._encoder_logstd(x, edge_index, *args, **kwargs)

    # noinspection PyTypedDict
    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        constructor_params = {
            "encoder_logstd_given": self.__encoder_logstd_given,
            "shared_encoder_given": self.__shared_encoder_given,
            "standalone": self.__standalone
        }

        # Encoder logstd params and weights
        constructor_params["encoder_logstd"] = {
            "state_dict": self._encoder_logstd.state_dict(),
            "constructor_params": self._encoder_logstd.serialize_constructor_params()
        }

        # Encoder mu params and weights
        constructor_params["encoder_mu"] = {
            "state_dict": self._encoder_mu.state_dict(),
            "constructor_params": self._encoder_mu.serialize_constructor_params()
        }

        # Shared encoder params
        if self.__shared_encoder_given:
            constructor_params["shared_encoder"] = {
                "state_dict": self._shared_encoder.state_dict(),
                "constructor_params": self._shared_encoder.serialize_constructor_params()
            }

        return constructor_params

    def reset_parameters(self):
        reset(self._encoder_mu)
        if self._encoder_logstd is not None:
            reset(self._encoder_logstd)
        if self._shared_encoder is not None:
            reset(self._shared_encoder)

    # noinspection PyMethodOverriding
    @classmethod
    def from_constructor_params(cls,
                                constructor_params: dict[str, Any],
                                encoder_mu_constructor: Type[SerializableModule],
                                encoder_logstd_constructor: Optional[Type[SerializableModule]] = None,
                                shared_encoder_constructor: Optional[Type[SerializableModule]] = None,
                                *args, **kwargs):

        try:
            standalone = constructor_params["standalone"]
        except KeyError:
            standalone = False

        # If encoder_logstd constructor is not given, its class will be the same as encoder_mu
        if not constructor_params["encoder_logstd_given"]:
            encoder_logstd_constructor = encoder_mu_constructor

        # Get mu encoder constructor params/state dict and construct it
        enc_mu_state_dict = constructor_params["encoder_mu"]["state_dict"]
        enc_mu_constructor_params = constructor_params["encoder_mu"]["constructor_params"]
        encoder_mu = encoder_mu_constructor.from_constructor_params(enc_mu_constructor_params)
        encoder_mu.load_state_dict(state_dict=enc_mu_state_dict)  # set weights

        # Get logstd encoder constructor params/state dict and construct it
        enc_logstd_state_dict = constructor_params["encoder_logstd"]["state_dict"]
        enc_logstd_constructor_params = constructor_params["encoder_logstd"]["constructor_params"]
        encoder_logstd = encoder_logstd_constructor.from_constructor_params(enc_logstd_constructor_params)
        encoder_logstd.load_state_dict(state_dict=enc_logstd_state_dict)  # set weights

        # Get shared encoder parameters/state dict and construct it if required
        shared_encoder = None
        if constructor_params["shared_encoder_given"]:
            shared_enc_state_dict = constructor_params["shared_encoder"]["state_dict"]
            shared_enc_constructor_params = constructor_params["shared_encoder"]["constructor_params"]
            shared_encoder = shared_encoder_constructor.from_constructor_params(shared_enc_constructor_params)
            shared_encoder.load_state_dict(state_dict=shared_enc_state_dict)  # set weights

        return cls(encoder_mu=encoder_mu, encoder_logstd=encoder_logstd, shared_encoder=shared_encoder,
                   standalone=standalone)


class VGAEv2(VGAE, SerializableModule):
    def __init__(self, encoder: VGEncoder, decoder: Optional[GraphDecoder] = None):
        """
        VGAE sub-class with a simple forward function implemented.

        :param encoder: The encoder network, which has to implement a forward method returning mu and logstd.
        :type encoder: torch.nn.Module
        :param decoder: Optional[torch.nn.Module] = None
        :type decoder: Optional[torch.nn.Module]
        """
        super(VGAEv2, self).__init__(encoder=encoder, decoder=decoder)
        self.__serialize_decoder = decoder is not None  # True if decoder is not None, False otherwise

    def forward(self, x, edge_index, sigmoid: bool = True, *args, **kwargs):

        mu, log_std = self.encoder(x, edge_index, *args, **kwargs)
        z = self.reparametrize(mu=mu, logstd=log_std)
        adj_rec = self.decode(z, edge_index, sigmoid=sigmoid)
        return adj_rec, mu, log_std

    def forward_all(self, x, edge_index, sigmoid: bool = True, decoder_kwargs: Optional[dict] = None, *args, **kwargs):
        """
        Takes the node features and corresponding edges, reconstructing a probabilistic adjacency matrix.

        :return a probabilistic adjacency matrix for the given input.
        :param x: node feature tensor
        :param edge_index: edge information.
        :param sigmoid: whether to apply a sigmoid function on the final decoder output, normalizing it.
        :type sigmoid: bool
        :param decoder_kwargs: dictionary of keyword arguments for the decoder.
        :type decoder_kwargs: dict

        :return: a probabilistic adjacency matrix for the given input.
        """
        mu, log_std = self.encoder(x, edge_index, *args, **kwargs)
        z = self.reparametrize(mu=mu, logstd=log_std)

        if decoder_kwargs is not None:
            adj_rec = self.decoder.forward_all(z, sigmoid=sigmoid, **decoder_kwargs)
        else:
            adj_rec = self.decoder.forward_all(z, sigmoid=sigmoid)
        return adj_rec

    # noinspection PyTypedDict
    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        constructor_params = {}

        # Serialize encoder
        constructor_params["encoder"] = {
            "state_dict": self.encoder.state_dict(),
            "constructor_params": self.encoder.serialize_constructor_params()
        }

        constructor_params["decoder"] = None
        # Serialize decoder if required
        if self.__serialize_decoder:
            constructor_params["decoder"] = {
                "state_dict": self.decoder.state_dict(),
                "constructor_params": self.decoder.serialize_constructor_params()
            }

        return constructor_params

    # noinspection PyMethodOverriding
    @classmethod
    def from_constructor_params(cls,
                                constructor_params: dict,
                                vgencoder_constructor: Type[VGEncoder],
                                encoder_mu_constructor: Type[SerializableModule],
                                encoder_logstd_constructor: Optional[Type[SerializableModule]] = None,
                                shared_encoder_constructor: Optional[Type[SerializableModule]] = None,
                                decoder_constructor: Optional[Type[GraphDecoder]] = None,
                                *args, **kwargs):

        # Get encoder constructor params/state dict and construct it
        enc_state_dict = constructor_params["encoder"]["state_dict"]
        enc_constructor_params = constructor_params["encoder"]["constructor_params"]
        encoder = vgencoder_constructor.from_constructor_params(
            enc_constructor_params,
            encoder_mu_constructor=encoder_mu_constructor,
            encoder_logstd_constructor=encoder_logstd_constructor,
            shared_encoder_constructor=shared_encoder_constructor
        )  # construct vgencoder
        encoder.load_state_dict(state_dict=enc_state_dict)  # set weights

        # If required, get decoder constructor params/state dict and construct it
        decoder = None
        if constructor_params["decoder"] is not None:
            dec_state_dict = constructor_params["decoder"]["state_dict"]
            dec_constructor_params = constructor_params["decoder"]["constructor_params"]
            decoder = decoder_constructor.from_constructor_params(dec_constructor_params)  # construct decoder
            decoder.load_state_dict(state_dict=dec_state_dict)  # set weights

        return cls(encoder=encoder, decoder=decoder)


def train_step_vgae(model: VGAEv2, train_data: DataLoader, optimizer, device, use_edge_weight: bool = False,
                    use_edge_attr: bool = False, forward_batch: bool = False):
    model.train()  # put the model in training mode

    running_loss = 0.0  # running average loss over the batches
    steps: int = 1

    for data in iter(train_data):
        data = data.to(device)  # move batch to device
        optimizer.zero_grad()  # reset the optimizer gradients

        if forward_batch:
            # Encoder output
            if use_edge_weight and use_edge_attr:
                z = model.encode(data.x, data.edge_index, edge_attr=data.edge_attr, edge_weight=data.edge_weight,
                                 batch=data.batch)
            elif use_edge_attr:
                z = model.encode(data.x, data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
            elif use_edge_weight:
                z = model.encode(data.x, data.edge_index, edge_weight=data.edge_weight, batch=data.batch)
            else:
                z = model.encode(data.x, data.edge_index, batch=data.batch)

        else:
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
        kl_divergence_loss = (1 / data.num_nodes) * model.kl_loss()  # KL-divergence loss, should work as mean on nodes
        loss = loss + kl_divergence_loss

        loss.backward()  # gradient update
        optimizer.step()  # advance the optimizer state

        # Update running average loss
        running_loss = running_loss + 1 / steps * (loss.item() - running_loss)

        print(f"Steps: {steps}/{len(train_data)}, running loss {running_loss}")
        steps += 1

    return float(running_loss)


@torch.no_grad()
def test_step_vgae(model: VGAEv2, val_data: DataLoader, device, use_edge_weight: bool = False,
                   use_edge_attr: bool = False, forward_batch: bool = False):
    model.eval()  # put the model in evaluation mode

    # Running average for loss, precision and AUC
    running_val_loss = 0.0
    running_auc = 0.0
    running_precision = 0.0
    steps: int = 1

    for data in iter(val_data):
        data = data.to(device)  # move batch to device

        if forward_batch:
            # Encoder output
            if use_edge_weight and use_edge_attr:
                z = model.encode(data.x, data.edge_index, edge_attr=data.edge_attr, edge_weight=data.edge_weight,
                                 batch=data.batch)
            elif use_edge_attr:
                z = model.encode(data.x, data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
            elif use_edge_weight:
                z = model.encode(data.x, data.edge_index, edge_weight=data.edge_weight, batch=data.batch)
            else:
                z = model.encode(data.x, data.edge_index, batch=data.batch)
        else:
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
        kl_divergence_loss = (1 / data.num_nodes) * model.kl_loss()  # KL-divergence loss, should work as mean on nodes
        loss = loss + kl_divergence_loss
        running_val_loss = running_val_loss + 1 / steps * (
                loss.item() - running_val_loss)  # update loss running average

        # Update AUC and precision running averages
        neg_edge_index = negative_sampling(data.edge_index, z.size(0))
        auc, avg_precision = model.test(z, pos_edge_index=data.edge_index, neg_edge_index=neg_edge_index)
        running_auc = running_auc + 1 / steps * (auc - running_auc)
        running_precision = running_precision + 1 / steps * (avg_precision - running_precision)
        steps += 1

    return float(running_val_loss), running_auc, running_precision


def train_vgae(model: VGAEv2, train_data: DataLoader, val_data: DataLoader, epochs: int, optimizer,
               experiment_path: str, experiment_name: str, use_edge_weight: bool = False, use_edge_attr: bool = False,
               early_stopping_patience: int = EARLY_STOP_PATIENCE, early_stopping_delta: float = 0,
               forward_batch: bool = False, tensorboard_log: bool = False) -> VGAEv2:
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    experiment_path = os.path.join(experiment_path, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)  # create experiment directory if it doesn't exist

    # Instantiate the summary writer
    writer = None
    if tensorboard_log:
        writer = SummaryWriter(f'{experiment_path}_{epochs}_epochs')

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

    # Do validation step
    val_loss, auc, avg_precision = test_step_vgae(
        model=model,
        val_data=val_data,
        device=device,
        use_edge_weight=use_edge_weight,
        use_edge_attr=use_edge_attr,
        forward_batch=forward_batch
    )
    # Check for early-stopping stuff
    monitor(val_loss, model)

    print('Epoch: {:d} Validation loss {:.4f}, AUC: {:.4f}, Average precision: {:.4f}'.format(0, val_loss, auc,
                                                                                              avg_precision))
    # Tensorboard state update
    if tensorboard_log:
        writer.add_scalar('auc_val', auc, 0)  # new line
        writer.add_scalar('val_loss', val_loss, 0)
        writer.add_scalar('avg_precision_val', avg_precision, 0)  # new line

    for epoch in range(0, epochs):
        # Do train step
        train_loss = train_step_vgae(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            device=device,
            use_edge_weight=use_edge_weight,
            use_edge_attr=use_edge_attr,
            forward_batch=forward_batch
        )

        # Do validation step
        val_loss, auc, avg_precision = test_step_vgae(
            model=model,
            val_data=val_data,
            device=device,
            use_edge_weight=use_edge_weight,
            use_edge_attr=use_edge_attr,
            forward_batch=forward_batch
        )

        print('Epoch: {:d}, Train loss: {:.4f}, Validation loss {:.4f}, '
              'AUC: {:.4f}, Average precision: {:.4f}'.format(epoch + 1, train_loss, val_loss, auc, avg_precision))

        # Tensorboard state update
        if tensorboard_log:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('auc_val', auc, epoch + 1)  # new line
            writer.add_scalar('val_loss', val_loss, epoch + 1)
            writer.add_scalar('avg_precision_val', avg_precision, epoch + 1)  # new line

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
        traced_max_metric='auc_val',
        store_path=os.path.join(f"{experiment_path}", "auc.svg")
    )

    mht.plot_metrics(
        ['avg_precision_val'],
        figsize=FIGURE_SIZE_DEFAULT,
        traced_max_metric='avg_precision_val',
        store_path=os.path.join(f"{experiment_path}", "avg_precision.svg")
    )

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))

    return model
