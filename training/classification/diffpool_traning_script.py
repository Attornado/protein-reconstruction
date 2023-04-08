import os
from typing import final
import torch
from torch_geometric.loader import DataLoader
from models.classification.classifiers import train_classifier
from models.classification.diffpool import DiffPool, DiffPoolMulticlassClassificationLoss
from preprocessing.constants import PSCDB_CLEANED_TRAIN, PSCDB_CLEANED_VAL, DATA_PATH, \
    PSCDB_CLASS_WEIGHTS
from preprocessing.dataset.dataset_creation import load_dataset
from log.logger import Logger
from torch.optim import Adam, Adadelta
import torchinfo


BATCH_SIZE: final = 20
EPOCHS: final = 3000
EARLY_STOPPING_PATIENCE: final = 200
EXPERIMENT_NAME: final = 'diffpool_test8'
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
    in_channels = 10
    n_classes = len(class_weights)
    l2 = 0.0  # try 5e-4
    learning_rate = 0.00001  # try 0.00001, 0.0001, 0.001
    optim = "adam"
    config = {
        "num_layers": 2,
        'dim_embedding': 128,
        'gnn_dim_hidden': 64,
        'dim_embedding_MLP': 50,
        "max_num_nodes": 3787
    }

    try:
        path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME, "best_acc.pt")
        best_model_acc = torch.load(path)["best_acc"]
    except FileNotFoundError:
        best_model_acc = -1

    # best_model_acc = 0.24839743971824646  # was -1
    print(f"Loaded best_model_acc {best_model_acc}")
    best_conf = None
    best_lr = None
    conf_count = 288

    grid_values = {
        "num_layers": [5],
        'dim_embedding': [128, 256],
        'gnn_dim_hidden': [64, 128, 256],
        'dim_embedding_MLP': [50, 100],
        "learning_rate": [0.0001, 0.00001, 0.000001]
    }

    for n in grid_values['num_layers']:
        for e in grid_values['dim_embedding']:
            for g in grid_values['gnn_dim_hidden']:
                for d in grid_values['dim_embedding_MLP']:
                    for lr in grid_values['learning_rate']:
                        config = {
                            'num_layers': n,
                            'dim_embedding': e,
                            'gnn_dim_hidden': g,
                            'dim_embedding_MLP': d,
                            "max_num_nodes": 3787
                        }

                        learning_rate = lr
                        sage = DiffPool(dim_features=in_channels, dim_target=n_classes, config=config)

                        if l2 > 0:
                            optimizer = Adam(sage.parameters(), lr=learning_rate, weight_decay=l2)
                        elif optim == "adam":
                            optimizer = Adam(sage.parameters(), lr=learning_rate)
                        else:
                            optimizer = Adadelta(sage.parameters())

                        conf_count += 1
                        full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME, f"n_{conf_count}")
                        checkpoint_path = os.path.join(full_experiment_path, "checkpoint.pt")
                        full_state_dict_path = os.path.join(full_experiment_path, "state_dict.pt")
                        if RESTORE_CHECKPOINT and os.path.exists(checkpoint_path):
                            print("Checkpoint found, loading state dict from checkpoint...")
                            state_dict = torch.load(checkpoint_path)
                            sage.load_state_dict(state_dict)
                            print("State dict loaded.")
                        elif RESTORE_CHECKPOINT and os.path.exists(full_state_dict_path):
                            print("Final state dict found, loading state dict...")
                            state_dict = torch.load(full_state_dict_path)
                            sage.load_state_dict(state_dict)
                            print("State dict loaded.")

                        print(sage)
                        print(torchinfo.summary(sage, depth=5))

                        full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME, f"n_{conf_count}")
                        logger = Logger(filepath=os.path.join(full_experiment_path, "trainlog.txt"), mode="a")
                        if not USE_CLASS_WEIGHTS:
                            class_weights = None  # set class weights to None if not use class weights is selected
                        else:
                            class_weights = class_weights.to(
                                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                            )
                        logger.log(f"Launching training for experiment DiffPool with config: \n {config} with learning "
                                   f"rate {lr}, \n stored in "
                                   f"{full_experiment_path}...")

                        model, metrics = train_classifier(
                            sage,
                            train_data=dl_train,
                            val_data=dl_val,
                            epochs=EPOCHS,
                            optimizer=optimizer,
                            experiment_path=EXPERIMENT_PATH,
                            experiment_name=os.path.join(EXPERIMENT_NAME, f"n_{conf_count}"),
                            early_stopping_patience=EARLY_STOPPING_PATIENCE,
                            criterion=DiffPoolMulticlassClassificationLoss(weights=class_weights,
                                                                           label_smoothing=LABEL_SMOOTHING),
                            logger=logger
                        )

                        if best_model_acc < metrics['accuracy']:
                            full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME)
                            logger = Logger(filepath=os.path.join(full_experiment_path, "trainlog.txt"), mode="a")
                            logger.log(f"Found better model than {best_model_acc} acc, with accuracy "
                                       f"{metrics['accuracy']} acc, saving it in best dir")
                            best_model_acc = metrics['accuracy']
                            best_conf = config
                            best_lr = lr
                            constructor_params = model.serialize_constructor_params()
                            state_dict = model.state_dict()
                            torch.save(state_dict, os.path.join(full_experiment_path, "state_dict.pt"))
                            torch.save(constructor_params, os.path.join(full_experiment_path, "constructor_params.pt"))
                            torch.save({"best_acc": best_model_acc},
                                       os.path.join(full_experiment_path, "best_acc.pt"))
                            logger.log(f"Model with lr {lr} and config {config} \n trained and stored to "
                                       f" {full_experiment_path}.")
                        del model

    '''
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
    '''


if __name__ == '__main__':
    main()
