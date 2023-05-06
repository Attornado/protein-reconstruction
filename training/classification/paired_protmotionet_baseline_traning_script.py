import os
import random
from typing import final
import torch
from torch_geometric.loader import DataLoader
from log.logger import Logger
from models.classification.classifiers import train_classifier, MulticlassClassificationLoss
from models.classification.protmotionnet import PairedProtMotionNet, train_paired_classifier
from models.pretraining.encoders import RevGCNEncoder, RevGATConvEncoder, RevSAGEConvEncoder, ResGCN2ConvEncoderV2
from models.classification.ugformer import GCN, GAT, SAGE
from preprocessing.constants import PSCDB_CLEANED_TRAIN, PSCDB_CLEANED_VAL, DATA_PATH, \
    PSCDB_CLASS_WEIGHTS, PSCDB_PAIRED_CLASS_WEIGHTS, PSCDB_PAIRED_CLEANED_TRAIN, PSCDB_PAIRED_CLEANED_VAL, \
    PSCDB_PAIRED_CLEANED_TEST, RANDOM_SEED
from preprocessing.dataset.dataset_creation import load_dataset
from torch.optim import Adam, Adadelta
import torchinfo

from preprocessing.dataset.paired_dataset import PairedDataLoader

BATCH_SIZE: final = 10
EPOCHS: final = 1000
EARLY_STOPPING_PATIENCE: final = 25
EXPERIMENT_NAME: final = 'paired_protmotionnet_test0'
EXPERIMENT_PATH: final = os.path.join(DATA_PATH, "fitted", "classification", "paired_protmotionnet")
RESTORE_CHECKPOINT: final = True
USE_CLASS_WEIGHTS: final = True
LABEL_SMOOTHING: final = 0.0
IN_CHANNELS: final = 10
CONF_COUNT_START: final = 0


def main():
    random.seed(RANDOM_SEED)
    ds_train = load_dataset(PSCDB_PAIRED_CLEANED_TRAIN, dataset_type="pscdb_paired")
    ds_val = load_dataset(PSCDB_PAIRED_CLEANED_VAL, dataset_type="pscdb_paired")
    # ds_test = load_dataset(PSCDB_CLEANED_TEST, dataset_type="pscdb")

    dl_train = PairedDataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = PairedDataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True)
    # dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)

    class_weights = torch.load(PSCDB_PAIRED_CLASS_WEIGHTS)
    in_channels = IN_CHANNELS
    n_classes = len(class_weights)
    l2 = 0.0  # try 5e-4
    optim = "adam"

    try:
        path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME, "best_acc.pt")
        best_model_acc = torch.load(path)["best_acc"]
    except FileNotFoundError:
        best_model_acc = -1

    # best_model_acc = 0.24839743971824646  # was -1
    print(f"Loaded best_model_acc {best_model_acc}")
    best_conf = None
    best_lr = None
    conf_count = 0

    grid_values = {
        'dropout': [0.3],
        "model_name": [GCN, SAGE, GAT],  # had GCN_MODEL_TYPE, GAT_MODEL_TYPE
        'embedding_dim': [32, 64, 128, 256],
        'n_heads_gat': [4],
        "dense_num": [2, 3],
        "n_layers": [1, 5, 20, 50, 100],
        "learning_rate": [0.0001, 0.000001, 0.0000001]
    }

    for m in grid_values['model_name']:
        for nl in grid_values['n_layers']:
            for emb in grid_values['embedding_dim']:
                for dn in grid_values['dense_num']:
                    for d in grid_values['dropout']:
                        for lr in grid_values['learning_rate']:
                            for nh in grid_values['n_heads_gat'] if m == GAT else [1]:
                                d = random.choice([d, d, d, 0.4, 0.2, 0.1])
                                if nl == 50:
                                    nl = random.choice([nl, 80])
                                nh = random.choice([nh, 8])
                                config = {
                                    'dropout': d,
                                    "model_name": m,
                                    "n_layers": nl,
                                    "embedding_dim": emb,
                                    "dense_num": dn,
                                    'learning_rate': lr
                                }

                                if m == GAT:
                                    config["n_heads_gat"] = nh

                                if conf_count < CONF_COUNT_START:
                                    conf_count += 1
                                    print(config)

                                else:
                                    learning_rate = lr
                                    encoder = None
                                    if m == GAT:
                                        encoder = RevGATConvEncoder(
                                            in_channels=in_channels,
                                            hidden_channels=emb,
                                            out_channels=emb,
                                            heads=nh,
                                            num_convs=nl,
                                            dropout=d,
                                            num_groups=2
                                        )
                                    elif m == SAGE:
                                        encoder = RevSAGEConvEncoder(
                                            in_channels=in_channels,
                                            hidden_channels=emb,
                                            out_channels=emb,
                                            num_convs=nl,
                                            dropout=d,
                                            num_groups=2
                                        )
                                    elif m == GCN:
                                        encoder = RevGCNEncoder(
                                            in_channels=in_channels,
                                            hidden_channels=emb,
                                            out_channels=emb,
                                            num_convs=nl,
                                            dropout=d,
                                            improved=random.choice([True, False]),
                                            num_groups=2
                                        )
                                    model = PairedProtMotionNet(
                                        encoder=encoder,
                                        encoder_out_channels=emb,
                                        dense_units=[emb, n_classes] if dn == 2 else [emb, emb/2, n_classes],
                                        dense_activations=["gelu", "linear"] if dn == 2 else ["gelu", "gelu", "linear"],
                                        dim_features=in_channels,
                                        dropout=d,
                                        readout=random.choice(['add_pool', "mean_pool"]),
                                        num_heads=2
                                    )
                                    if l2 > 0:
                                        optimizer = Adam(model.parameters(), lr=learning_rate,
                                                         weight_decay=l2)
                                    elif optim == "adam":
                                        optimizer = Adam(model.parameters(), lr=learning_rate)
                                    else:
                                        optimizer = Adadelta(model.parameters())

                                    conf_count += 1
                                    full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME,
                                                                        f"n_{conf_count}")
                                    checkpoint_path = os.path.join(full_experiment_path, "checkpoint.pt")
                                    full_state_dict_path = os.path.join(full_experiment_path, "state_dict.pt")
                                    if RESTORE_CHECKPOINT and os.path.exists(checkpoint_path):
                                        print("Checkpoint found, loading state dict from checkpoint...")
                                        state_dict = torch.load(checkpoint_path)
                                        model.load_state_dict(state_dict)
                                        print("State dict loaded.")
                                    elif RESTORE_CHECKPOINT and os.path.exists(full_state_dict_path):
                                        print("Final state dict found, loading state dict...")
                                        state_dict = torch.load(full_state_dict_path)
                                        model.load_state_dict(state_dict)
                                        print("State dict loaded.")

                                    print(model)
                                    print(torchinfo.summary(model, depth=5))

                                    full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME,
                                                                        f"n_{conf_count}")
                                    logger = Logger(filepath=os.path.join(full_experiment_path, "trainlog.txt"),
                                                    mode="a")
                                    if not USE_CLASS_WEIGHTS:
                                        class_weights = None
                                        # set class weights to None if not use class weights is selected
                                    else:
                                        class_weights = class_weights.to(
                                            torch.device("cuda") if torch.cuda.is_available() else
                                            torch.device("cpu")
                                        )
                                    logger.log(f"Launching training for experiment UGFormerV2 n{conf_count} "
                                               f"with config \n {config} with learning rate "
                                               f"{lr}, \n stored in "
                                               f"{full_experiment_path}...")

                                    model, metrics = train_paired_classifier(
                                        model,
                                        train_data=dl_train,
                                        val_data=dl_val,
                                        epochs=EPOCHS,
                                        optimizer=optimizer,
                                        experiment_path=EXPERIMENT_PATH,
                                        experiment_name=os.path.join(EXPERIMENT_NAME, f"n_{conf_count}"),
                                        early_stopping_patience=EARLY_STOPPING_PATIENCE,
                                        criterion=MulticlassClassificationLoss(weights=class_weights,
                                                                               label_smoothing=LABEL_SMOOTHING),
                                        logger=logger
                                    )

                                    if best_model_acc < metrics['accuracy']:
                                        full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME)
                                        logger = Logger(filepath=os.path.join(full_experiment_path, "trainlog.txt"),
                                                        mode="a")
                                        logger.log(f"Found better model n{conf_count} than {best_model_acc} acc, "
                                                   f"with accuracy "
                                                   f"{metrics['accuracy']} acc, saving it in best dir")
                                        best_model_acc = metrics['accuracy']
                                        best_conf = config
                                        best_lr = lr
                                        constructor_params = model.serialize_constructor_params()
                                        state_dict = model.state_dict()
                                        torch.save(state_dict, os.path.join(full_experiment_path,
                                                                            "state_dict.pt"))
                                        torch.save(constructor_params, os.path.join(full_experiment_path,
                                                                                    "constructor_params.pt"))
                                        torch.save({"best_acc": best_model_acc},
                                                   os.path.join(full_experiment_path, "best_acc.pt"))
                                        logger.log(f"Model with lr {lr} and config {config} \n trained and "
                                                   f"stored to {full_experiment_path}.")
                                    del model


if __name__ == '__main__':
    main()
