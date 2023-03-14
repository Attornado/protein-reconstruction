import os
from typing import final
import torch
from torch_geometric.loader import DataLoader
from models.classification.dgcnn import DGCNN
from models.classification.classifiers import train_classifier
from preprocessing.constants import PSCDB_CLEANED_TRAIN, PSCDB_CLEANED_VAL, PSCDB_CLEANED_TEST, DATA_PATH
from preprocessing.dataset import load_dataset
from torch.optim import Adam, Adadelta
import torchinfo


BATCH_SIZE: final = 200
EPOCHS: final = 1000
EARLY_STOPPING_PATIENCE: final = 50
EXPERIMENT_NAME: final = 'dgcnn_test5'
EXPERIMENT_PATH: final = os.path.join(DATA_PATH, "fitted", "classification", "dgcnn")
RESTORE_CHECKPOINT: final = True


def main():
    ds_train = load_dataset(PSCDB_CLEANED_TRAIN, dataset_type="pscdb")
    ds_val = load_dataset(PSCDB_CLEANED_VAL, dataset_type="pscdb")
    # ds_test = load_dataset(PSCDB_CLEANED_TEST, dataset_type="pscdb")

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True)
    # dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)

    in_channels = 10
    n_classes = 7
    l2 = 0.0  # try 5e-4
    learning_rate = 0.00001  # try 0.0001
    optim = "adam"
    config = {
        "num_layers": 4,
        "embedding_dim": 64,
        "dense_dim": 128,
        "dataset": "PSCDB",
        "k": 0.9
    }

    dgcnn = DGCNN(dim_features=in_channels, dim_target=n_classes, config=config)

    full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME)
    checkpoint_path = os.path.join(full_experiment_path, "checkpoint.pt")
    full_state_dict_path = os.path.join(full_experiment_path, "state_dict.pt")
    if RESTORE_CHECKPOINT and os.path.exists(checkpoint_path):
        print("Checkpoint found, loading state dict from checkpoint...")
        state_dict = torch.load(checkpoint_path)
        dgcnn.load_state_dict(state_dict)
        print("State dict loaded.")
    elif RESTORE_CHECKPOINT and os.path.exists(full_state_dict_path):
        print("Final state dict found, loading state dict...")
        state_dict = torch.load(full_state_dict_path)
        dgcnn.load_state_dict(state_dict)
        print("State dict loaded.")

    print(dgcnn)
    print(torchinfo.summary(dgcnn, depth=5))

    if l2 > 0:
        optimizer = Adam(dgcnn.parameters(), lr=learning_rate, weight_decay=l2)
    elif optim == "adam":
        optimizer = Adam(dgcnn.parameters(), lr=learning_rate)
    else:
        optimizer = Adadelta(dgcnn.parameters())
    model = train_classifier(
        dgcnn,
        train_data=dl_train,
        val_data=dl_val,
        epochs=EPOCHS,
        optimizer=optimizer,
        experiment_path=EXPERIMENT_PATH,
        experiment_name=EXPERIMENT_NAME,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )

    full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME)
    constructor_params = model.serialize_constructor_params()
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(full_experiment_path, "state_dict.pt"))
    torch.save(constructor_params, os.path.join(full_experiment_path, "constructor_params.pt"))
    print(f"Model trained and stored to {full_experiment_path}.")


if __name__ == '__main__':
    main()
