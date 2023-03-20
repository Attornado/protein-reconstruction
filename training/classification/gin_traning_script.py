import os
from typing import final
import torch
from torch_geometric.loader import DataLoader
from log.logger import Logger
from models.classification.gin import GIN
from models.classification.classifiers import train_classifier, MulticlassClassificationLoss
from preprocessing.constants import PSCDB_CLEANED_TRAIN, PSCDB_CLEANED_VAL, PSCDB_CLEANED_TEST, DATA_PATH, \
    PSCDB_CLASS_WEIGHTS
from preprocessing.dataset import load_dataset
from torch.optim import Adam, Adadelta
import torchinfo


BATCH_SIZE: final = 200
EPOCHS: final = 1000
EARLY_STOPPING_PATIENCE: final = 50
EXPERIMENT_NAME: final = 'gin_test0'
EXPERIMENT_PATH: final = os.path.join(DATA_PATH, "fitted", "classification", "gin")
RESTORE_CHECKPOINT: final = False
USE_CLASS_WEIGHTS: final = True
LABEL_SMOOTHING: final = 0.0


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
    learning_rate = 0.01  # 0.0001, 0.01, 0.00001
    optim = "adam"
    config = {
        'dropout': 0.5,  # 0.1, 0.5, 0.0
        'hidden_units': [64, 64, 64, 64],  # [64, 64, 64, 64], [32, 32, 32, 32], [64], [32, 32]
        "train_eps": True,  # try True, False
        'aggregation': "sum",  # try 'mean', 'sum'
    }

    gin = GIN(dim_features=in_channels, dim_target=n_classes, config=config)

    full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME)
    checkpoint_path = os.path.join(full_experiment_path, "checkpoint.pt")
    full_state_dict_path = os.path.join(full_experiment_path, "state_dict.pt")
    if RESTORE_CHECKPOINT and os.path.exists(checkpoint_path):
        print("Checkpoint found, loading state dict from checkpoint...")
        state_dict = torch.load(checkpoint_path)
        gin.load_state_dict(state_dict)
        print("State dict loaded.")
    elif RESTORE_CHECKPOINT and os.path.exists(full_state_dict_path):
        print("Final state dict found, loading state dict...")
        state_dict = torch.load(full_state_dict_path)
        gin.load_state_dict(state_dict)
        print("State dict loaded.")

    print(gin)
    print(torchinfo.summary(gin, depth=5))

    if l2 > 0:
        optimizer = Adam(gin.parameters(), lr=learning_rate, weight_decay=l2)
    elif optim == "adam":
        optimizer = Adam(gin.parameters(), lr=learning_rate)
    else:
        optimizer = Adadelta(gin.parameters())
    full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME)
    logger = Logger(filepath=os.path.join(full_experiment_path, "trainlog.txt"), mode="a")
    if not USE_CLASS_WEIGHTS:
        class_weights = None  # set class weights to None if not use class weights is selected
    else:
        class_weights = class_weights.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    model = train_classifier(
        gin,
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
