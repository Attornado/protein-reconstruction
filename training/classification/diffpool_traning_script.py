import os
from typing import final
import torch
from torch_geometric.loader import DataLoader
from models.classification.classifiers import train_classifier
from models.classification.diffpool import DiffPool, DiffPoolMulticlassClassificationLoss
from preprocessing.constants import PSCDB_CLEANED_TRAIN, PSCDB_CLEANED_VAL, PSCDB_CLEANED_TEST, DATA_PATH, \
    PSCDB_CLASS_WEIGHTS
from preprocessing.dataset import load_dataset
from log.logger import Logger
from torch.optim import Adam, Adadelta
import torchinfo

BATCH_SIZE: final = 20
EPOCHS: final = 3000
EARLY_STOPPING_PATIENCE: final = 200
EXPERIMENT_NAME: final = 'diffpool_test7'
EXPERIMENT_PATH: final = os.path.join(DATA_PATH, "fitted", "classification", "diffpool")
RESTORE_CHECKPOINT: final = True
USE_CLASS_WEIGHTS: final = True
LABEL_SMOOTHING: final = 0.1


def main():
    ds_train = load_dataset(PSCDB_CLEANED_TRAIN, dataset_type="pscdb")
    ds_val = load_dataset(PSCDB_CLEANED_VAL, dataset_type="pscdb")
    # ds_test = load_dataset(PSCDB_CLEANED_TEST, dataset_type="pscdb")

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True)
    # dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)

    class_weights = torch.load(PSCDB_CLASS_WEIGHTS)
    in_channels = 30
    n_classes = len(class_weights)
    l2 = 0.0  # try 5e-4
    learning_rate = 0.00001  # try 0.00001, 0.0001, 0.001
    optim = "adadelta"
    config = {
        "num_layers": 2,
        'dim_embedding': 128,
        'gnn_dim_hidden': 64,
        'dim_embedding_MLP': 50,
        "max_num_nodes": 3787
    }

    diffpool = DiffPool(dim_features=in_channels, dim_target=n_classes, config=config)

    full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME)
    checkpoint_path = os.path.join(full_experiment_path, "checkpoint.pt")
    full_state_dict_path = os.path.join(full_experiment_path, "state_dict.pt")
    if RESTORE_CHECKPOINT and os.path.exists(checkpoint_path):
        print("Checkpoint found, loading state dict from checkpoint...")
        state_dict = torch.load(checkpoint_path)
        diffpool.load_state_dict(state_dict)
        print("State dict loaded.")
    elif RESTORE_CHECKPOINT and os.path.exists(full_state_dict_path):
        print("Final state dict found, loading state dict...")
        state_dict = torch.load(full_state_dict_path)
        diffpool.load_state_dict(state_dict)
        print("State dict loaded.")

    print(diffpool)
    print(torchinfo.summary(diffpool, depth=5))

    if l2 > 0:
        optimizer = Adam(diffpool.parameters(), lr=learning_rate, weight_decay=l2)
    elif optim == "adam":
        optimizer = Adam(diffpool.parameters(), lr=learning_rate)
    else:
        optimizer = Adadelta(diffpool.parameters())
    full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME)
    logger = Logger(filepath=os.path.join(full_experiment_path, "trainlog.txt"), mode="a")
    if not USE_CLASS_WEIGHTS:
        class_weights = None  # set class weights to None if not use class weights is selected
    else:
        class_weights = class_weights.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = train_classifier(
        diffpool,
        train_data=dl_train,
        val_data=dl_val,
        epochs=EPOCHS,
        optimizer=optimizer,
        experiment_path=EXPERIMENT_PATH,
        experiment_name=EXPERIMENT_NAME,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        criterion=DiffPoolMulticlassClassificationLoss(weights=class_weights, label_smoothing=LABEL_SMOOTHING),
        logger=logger
    )

    constructor_params = model.serialize_constructor_params()
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(full_experiment_path, "state_dict.pt"))
    torch.save(constructor_params, os.path.join(full_experiment_path, "constructor_params.pt"))
    logger.log(f"Model trained and stored to {full_experiment_path}.")
    # print()


if __name__ == '__main__':
    main()
