import os
from typing import final
import torch
from torch_geometric.loader import DataLoader
from log.logger import Logger
from models.classification.pscdb_baseline import GNNBaselinePSCDB, GAT_MODEL_TYPE
from models.classification.classifiers import train_classifier, MulticlassClassificationLoss
from preprocessing.constants import PSCDB_CLEANED_TRAIN, PSCDB_CLEANED_VAL, DATA_PATH, \
    PSCDB_CLASS_WEIGHTS
from preprocessing.dataset.dataset_creation import load_dataset
from torch.optim import Adam, Adadelta
import torchinfo


BATCH_SIZE: final = 50
EPOCHS: final = 2000
EARLY_STOPPING_PATIENCE: final = 300
EXPERIMENT_NAME: final = 'pscdb_baseline_test3'
EXPERIMENT_PATH: final = os.path.join(DATA_PATH, "fitted", "classification", "pscdb_baseline")
RESTORE_CHECKPOINT: final = False
USE_CLASS_WEIGHTS: final = True
LABEL_SMOOTHING: final = 0.0
IN_CHANNELS: final = 10


def main():
    ds_train = load_dataset(PSCDB_CLEANED_TRAIN, dataset_type="pscdb")
    ds_val = load_dataset(PSCDB_CLEANED_VAL, dataset_type="pscdb")
    # ds_test = load_dataset(PSCDB_CLEANED_TEST, dataset_type="pscdb")

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True)
    # dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)

    class_weights = torch.load(PSCDB_CLASS_WEIGHTS)
    in_channels = IN_CHANNELS
    n_classes = len(class_weights)
    l2 = 0.0  # try 5e-4
    learning_rate = 0.00001  # 0.001, 0.0001, 0.01, 0.00001
    optim = "adam"
    config = {
        'dropout': 0.5,  # 0.1, 0.5, 0.0
        "model_name": GAT_MODEL_TYPE,  # GCN_MODEL_TYPE, GAT_MODEL_TYPE, SAGE_MODEL_TYPE
        'hidden_dim': 100,  # try 10, 32, 64, 50, 100, 200
        'out_dim': 100,  # try 10, 20, 50, 100, 200,
        'n_heads': 32
    }

    pscdb_baseline = GNNBaselinePSCDB(dim_features=in_channels, dim_target=n_classes, config=config)

    full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME)
    checkpoint_path = os.path.join(full_experiment_path, "checkpoint.pt")
    full_state_dict_path = os.path.join(full_experiment_path, "state_dict.pt")
    if RESTORE_CHECKPOINT and os.path.exists(checkpoint_path):
        print("Checkpoint found, loading state dict from checkpoint...")
        state_dict = torch.load(checkpoint_path)
        pscdb_baseline.load_state_dict(state_dict)
        print("State dict loaded.")
    elif RESTORE_CHECKPOINT and os.path.exists(full_state_dict_path):
        print("Final state dict found, loading state dict...")
        state_dict = torch.load(full_state_dict_path)
        pscdb_baseline.load_state_dict(state_dict)
        print("State dict loaded.")

    print(pscdb_baseline)
    print(torchinfo.summary(pscdb_baseline, depth=5))

    if l2 > 0 and optim == "adam":
        optimizer = Adam(pscdb_baseline.parameters(), lr=learning_rate, weight_decay=l2)
    elif optim == "adam":
        optimizer = Adam(pscdb_baseline.parameters(), lr=learning_rate)
    else:
        optimizer = Adadelta(pscdb_baseline.parameters())
    full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME)
    logger = Logger(filepath=os.path.join(full_experiment_path, "trainlog.txt"), mode="a")
    if not USE_CLASS_WEIGHTS:
        class_weights = None  # set class weights to None if not use class weights is selected
    else:
        class_weights = class_weights.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    model = train_classifier(
        pscdb_baseline,
        train_data=dl_train,
        val_data=dl_val,
        epochs=EPOCHS,
        optimizer=optimizer,
        experiment_path=EXPERIMENT_PATH,
        experiment_name=EXPERIMENT_NAME,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        criterion=MulticlassClassificationLoss(weights=class_weights, label_smoothing=LABEL_SMOOTHING),
        logger=logger
    )

    constructor_params = model.serialize_constructor_params()
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(full_experiment_path, "state_dict.pt"))
    torch.save(constructor_params, os.path.join(full_experiment_path, "constructor_params.pt"))
    logger.log(f"Model trained and stored to {full_experiment_path}.")


if __name__ == '__main__':
    main()
