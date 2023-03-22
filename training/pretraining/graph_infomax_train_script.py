import os
from typing import final
import torch
from torch.optim import Adadelta
from torch_geometric.loader import DataLoader
from torchinfo import torchinfo
from models.pretraining.encoders import RevGCNEncoder
from models.pretraining.graph_infomax import DeepGraphInfomaxV2, train_DGI, MeanPoolReadout, \
    RandomPermutationCorruption
from preprocessing.constants import PRETRAIN_CLEANED_TRAIN, PRETRAIN_CLEANED_VAL, DATA_PATH
from preprocessing.dataset.dataset_creation import load_dataset


BATCH_SIZE: final = 500
EPOCHS: final = 150
EARLY_STOPPING_PATIENCE: final = 20
EXPERIMENT_NAME: final = 'dgi_rev_gcn_test14'
EXPERIMENT_PATH: final = os.path.join(DATA_PATH, "fitted", "pretraining", "dgi")
RESTORE_CHECKPOINT: final = True


def main():
    ds_train = load_dataset(PRETRAIN_CLEANED_TRAIN, dataset_type="pretrain")
    ds_val = load_dataset(PRETRAIN_CLEANED_VAL, dataset_type="pretrain")
    # ds_test = load_dataset(PRETRAIN_CLEANED_TEST, dataset_type="pretrain")

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True)
    # dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)

    dl_train_corruption = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val_corruption = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True)

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
    
     encoder = RevGCNEncoder(
        in_channels=in_channels,
        hidden_channels=30,
        out_channels=30,
        num_convs=6,  # was 30
        improved=False,
        dropout=0.0,
        num_groups=10,
        normalize_hidden=True
    )
    
    encoder = RevSAGEConvEncoder(
        in_channels=in_channels,
        hidden_channels=100,
        out_channels=100,
        num_convs=60,
        dropout=0.1,
        project=False,
        root_weight=True,
        aggr="mean",
        num_groups=5,
        normalize_hidden=True
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
    
    encoder = RevGATConvEncoder(
        in_channels=in_channels,
        hidden_channels=100,
        out_channels=100,
        num_convs=60,
        dropout=0.1,
        heads=5,
        concat=False,
        num_groups=5
    )
    
    encoder_mu = GATConvBlock(
        in_channels=30,
        out_channels=30,
        heads=4
    )
    """

    encoder = RevGCNEncoder(
        in_channels=in_channels,
        hidden_channels=150,  # was 100
        out_channels=150,  # was 100
        num_convs=60,
        improved=True,
        dropout=0.1,
        num_groups=5,
        normalize_hidden=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    readout = MeanPoolReadout(device=device, sigmoid=False)
    # corruption = RandomSampleCorruption(train_data=dl_train_corruption, val_data=dl_val_corruption, device=device)
    corruption = RandomPermutationCorruption(device=device)
    dgi = DeepGraphInfomaxV2(
        hidden_channels=150,  # was 100
        encoder=encoder,
        normalize_hidden=True,
        readout=readout,
        corruption=corruption,
        dropout=0.0
    )

    print(dgi)
    print(torchinfo.summary(dgi, depth=5))

    full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME)
    checkpoint_path = os.path.join(full_experiment_path, "checkpoint.pt")
    full_state_dict_path = os.path.join(full_experiment_path, "state_dict.pt")
    if RESTORE_CHECKPOINT and os.path.exists(checkpoint_path):
        print("Checkpoint found, loading state dict from checkpoint...")
        state_dict = torch.load(checkpoint_path)
        dgi.load_state_dict(state_dict)
        print("State dict loaded.")
    elif RESTORE_CHECKPOINT and os.path.exists(full_state_dict_path):
        print("Final state dict found, loading state dict...")
        state_dict = torch.load(full_state_dict_path)
        dgi.load_state_dict(state_dict)
        print("State dict loaded.")

    # optimizer = Adam(dgi.parameters(), lr=0.01)
    optimizer = Adadelta(dgi.parameters())
    model = train_DGI(
        dgi,
        train_data=dl_train,
        val_data=dl_val,
        epochs=EPOCHS,
        optimizer=optimizer,
        experiment_path=EXPERIMENT_PATH,
        experiment_name=EXPERIMENT_NAME,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )

    constructor_params = model.serialize_constructor_params()
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(full_experiment_path, "state_dict.pt"))
    torch.save(constructor_params, os.path.join(full_experiment_path, "constructor_params.pt"))
    print(f"Model trained and stored to {full_experiment_path}.")


if __name__ == "__main__":
    main()
