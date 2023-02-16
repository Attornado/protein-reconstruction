import os
from typing import final
import torch
from torch_geometric.loader import DataLoader
from models.layers import GATConvBlock, SAGEConvBlock, GCNConvBlock
from preprocessing.constants import PRETRAIN_CLEANED_TRAIN, PRETRAIN_CLEANED_VAL, PRETRAIN_CLEANED_TEST, DATA_PATH
from models.pretraining.vgae import VGAEv2, train_vgae, VGEncoder
from models.pretraining.encoders import RevGATConvEncoder, RevSAGEConvEncoder, ResGCN2ConvEncoder, SimpleGCNEncoder, \
    RevGCNEncoder
from preprocessing.dataset import load_dataset
from torch.optim import Adam, Adadelta
import torchinfo
from training.training_tools import EARLY_STOP_PATIENCE


BATCH_SIZE: final = 500
EPOCHS: final = 250
EXPERIMENT_NAME: final = 'vgae_rev_gcn_test11'
EXPERIMENT_PATH: final = os.path.join(DATA_PATH, "fitted", "pretraining", "vgae")


def main():
    ds_train = load_dataset(PRETRAIN_CLEANED_TRAIN, dataset_type="pretrain")
    ds_val = load_dataset(PRETRAIN_CLEANED_VAL, dataset_type="pretrain")
    # ds_test = load_dataset(PRETRAIN_CLEANED_TEST, dataset_type="pretrain")

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True)
    # dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)

    in_channels = 10

    """
    encoder = ResGCN2ConvEncoder(
        in_channels=in_channels,
        hidden_channels=10,
        out_channels=10,
        num_convs=5,
        dropout=0.0,
        alpha=0.4
    )

    encoder = SimpleGCNEncoder(
        in_channels=in_channels,
        hidden_channels=10,
        out_channels=10,
        conv_dims=[10, 10, 10, 10],
        dropout=0.0
    )
    
     encoder = RevSAGEConvEncoder(
        in_channels=in_channels,
        hidden_channels=40,
        out_channels=40,
        num_convs=60,
        dropout=0.0,
        project=False,
        root_weight=True,
        aggr="mean",
        num_groups=10,
        normalize_hidden=True
    )

    encoder_mu = SAGEConvBlock(
        in_channels=40,
        out_channels=40,
        project=False,
        root_weight=True,
        aggr="mean"
    )
    
     encoder = RevSAGEConvEncoder(
        in_channels=in_channels,
        hidden_channels=10,
        out_channels=10,
        num_convs=20,
        dropout=0.0,
        num_groups=2
    )   
    
    encoder = RevGATConvEncoder(
        in_channels=in_channels,
        hidden_channels=30,
        out_channels=30,
        num_convs=50,
        dropout=0.0,
        heads=4,
        concat=False,
        num_groups=10
    )
    
    encoder_mu = GATConvBlock(
        in_channels=30,
        out_channels=30,
        heads=4
    )
    """

    encoder = RevGCNEncoder(
        in_channels=in_channels,
        hidden_channels=30,
        out_channels=30,
        num_convs=50,
        improved=True,
        dropout=0.0,
        num_groups=10,
        normalize_hidden=True
    )

    encoder_mu = GCNConvBlock(
        in_channels=30,
        out_channels=30,
        improved=True
    )

    vgencoder = VGEncoder(shared_encoder=encoder, encoder_mu=encoder_mu)
    vgae = VGAEv2(encoder=vgencoder)
    print(vgae)
    print(torchinfo.summary(vgae))

    optimizer = Adam(vgae.parameters(), lr=0.1, weight_decay=5e-4)
    optimizer = Adadelta(vgae.parameters())
    model = train_vgae(
        vgae,
        train_data=dl_train,
        val_data=dl_val,
        epochs=EPOCHS,
        optimizer=optimizer,
        experiment_path=EXPERIMENT_PATH,
        experiment_name=EXPERIMENT_NAME,
        early_stopping_patience=30
    )

    full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME)
    constructor_params = model.serialize_constructor_params()
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(full_experiment_path, "state_dict.pt"))
    torch.save(constructor_params, os.path.join(full_experiment_path, "constructor_params.pt"))
    print(f"Model trained and stored to {full_experiment_path}.")


if __name__ == '__main__':
    main()
