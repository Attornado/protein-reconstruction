import os
from typing import final
import torch
from torch_geometric.loader import DataLoader
from models.classification.sage import SAGEClassifier
from models.layers import GCNConvBlock, GATConvBlock
from preprocessing.constants import PRETRAIN_CLEANED_TRAIN, PRETRAIN_CLEANED_VAL, DATA_PATH, RANDOM_SEED
from models.pretraining.vgae import VGAEv2, train_vgae, VGEncoder
from models.pretraining.encoders import RevGCNEncoder, RevGATConvEncoder
from preprocessing.dataset.dataset_creation import load_dataset
from torch.optim import Adam, Adadelta
import torchinfo
from training.training_tools import seed_everything


BATCH_SIZE: final = 160
EPOCHS: final = 250
EARLY_STOPPING_PATIENCE: final = 30
EXPERIMENT_NAME: final = 'vgae_rev_gcn_test19'
EXPERIMENT_PATH: final = os.path.join(DATA_PATH, "fitted", "pretraining", "vgae")
RESTORE_CHECKPOINT: final = True


def main():
    seed_everything(seed=RANDOM_SEED)
    ds_train = load_dataset(PRETRAIN_CLEANED_TRAIN, dataset_type="pretrain")
    ds_val = load_dataset(PRETRAIN_CLEANED_VAL, dataset_type="pretrain")
    # ds_test = load_dataset(PRETRAIN_CLEANED_TEST, dataset_type="pretrain")

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True)
    # dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)

    in_channels = 10

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

    encoder = RevSAGEConvEncoder(
        in_channels=in_channels,
        hidden_channels=100,
        out_channels=100,
        num_convs=60,
        dropout=0.0,
        project=False,
        root_weight=True,
        aggr="mean",
        num_groups=5,
        normalize_hidden=True
    )

    encoder_mu = SAGEConvBlock(
        in_channels=100,
        out_channels=100,
        project=False,
        root_weight=True,
        aggr="mean"
    )
    encoder = RevGATConvEncoder(
        in_channels=in_channels,
        hidden_channels=100,
        out_channels=100,
        num_convs=60,
        dropout=0.1,
        heads=5,
        concat=False,
        num_groups=5  # was 10, remember to change to 5 on the next experiment (14)
    )

    encoder_mu = GATConvBlock(
        in_channels=100,
        out_channels=100,
        heads=5
    )

    encoder = RevGCNEncoder(
        in_channels=in_channels,
        hidden_channels=100,
        out_channels=100,
        num_convs=60,
        improved=True,
        dropout=0.1,
        num_groups=5,
        normalize_hidden=True
    )
    """

    emb_dim = 128
    n_layers = 60

    encoder = RevGATConvEncoder(
        in_channels=in_channels,
        hidden_channels=emb_dim,
        out_channels=emb_dim,
        num_convs=n_layers,
        dropout=0.1,
        heads=5,
        concat=False,
        num_groups=2  # was 10, remember to change to 5 on the next experiment (14)
    )
    encoder_mu = GATConvBlock(
        in_channels=emb_dim,
        out_channels=emb_dim,
        heads=5,
        concat=True
    )
    '''encoder_mu = SAGEClassifier(dim_features=in_channels,
                                dim_target=7,
                                config={"dim_embedding": emb_dim,
                                        "num_layers": n_layers,
                                        "return_embeddings": True,
                                        'aggregation': 'mean'})'''

    '''encoder_mu = GCNConvBlock(
        in_channels=emb_dim,
        out_channels=emb_dim,
        improved=True
    )'''

    vgencoder = VGEncoder(shared_encoder=encoder, encoder_mu=encoder_mu)
    vgae = VGAEv2(encoder=vgencoder)

    full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME)
    checkpoint_path = os.path.join(full_experiment_path, "checkpoint.pt")
    full_state_dict_path = os.path.join(full_experiment_path, "state_dict.pt")
    if RESTORE_CHECKPOINT and os.path.exists(checkpoint_path):
        print("Checkpoint found, loading state dict from checkpoint...")
        state_dict = torch.load(checkpoint_path)
        vgae.load_state_dict(state_dict)
        print("State dict loaded.")
    elif RESTORE_CHECKPOINT and os.path.exists(full_state_dict_path):
        print("Final state dict found, loading state dict...")
        state_dict = torch.load(full_state_dict_path)
        vgae.load_state_dict(state_dict)
        print("State dict loaded.")

    print(vgae)
    print(torchinfo.summary(vgae, depth=6))

    # optimizer = Adam(vgae.parameters(), lr=0.1, weight_decay=5e-4)
    optimizer = Adadelta(vgae.parameters())
    model = train_vgae(
        vgae,
        train_data=dl_train,
        val_data=dl_val,
        epochs=EPOCHS,
        optimizer=optimizer,
        experiment_path=EXPERIMENT_PATH,
        experiment_name=EXPERIMENT_NAME,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        forward_batch=False
    )

    full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME)
    constructor_params = model.serialize_constructor_params()
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(full_experiment_path, "state_dict.pt"))
    torch.save(constructor_params, os.path.join(full_experiment_path, "constructor_params.pt"))
    print(f"Model trained and stored to {full_experiment_path}.")


if __name__ == '__main__':
    main()
