import os
from typing import final
import torch
from torch_geometric.loader import DataLoader
from log.logger import Logger
from models.classification.gin import GIN
from models.classification.classifiers import train_classifier, MulticlassClassificationLoss
from preprocessing.constants import PSCDB_CLEANED_TRAIN, PSCDB_CLEANED_VAL, DATA_PATH, \
    PSCDB_CLASS_WEIGHTS
from preprocessing.dataset.dataset_creation import load_dataset
from torch.optim import Adam, Adadelta
import torchinfo

BATCH_SIZE: final = 200
EPOCHS: final = 2000
EARLY_STOPPING_PATIENCE: final = 200
EXPERIMENT_NAME: final = 'gin_test0'
EXPERIMENT_PATH: final = os.path.join(DATA_PATH, "fitted", "classification", "gin")
RESTORE_CHECKPOINT: final = True
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
    in_channels = 10
    n_classes = len(class_weights)
    l2 = 5e-4
    learning_rate = 0.01  # 0.0001, 0.01, 0.00001
    optim = "adam"

    grid_values = {
        'dropout': [0.0, 0.1, 0.2, 0.5],
        'hidden_units': reversed([[128, 128, 128, 128], [64, 64, 64, 64], [32, 32, 32, 32], [64], [32, 32]]),
        'train_eps': [True, False],
        'aggregation': ["sum", "mean"],
        'learning_rate': [0.0001, 0.01, 0.00001, 0.001, 0.000001],
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
    conf_count = 0

    for d in grid_values['dropout']:
        for h in grid_values['hidden_units']:
            for t in grid_values['train_eps']:
                for a in grid_values['aggregation']:
                    for lr in grid_values['learning_rate']:
                        config = {
                            'dropout': d,
                            'hidden_units': h,
                            "train_eps": t,
                            'aggregation': a,
                        }

                        learning_rate = lr
                        gin = GIN(dim_features=in_channels, dim_target=n_classes, config=config)

                        if l2 > 0:
                            optimizer = Adam(gin.parameters(), lr=learning_rate, weight_decay=l2)
                        elif optim == "adam":
                            optimizer = Adam(gin.parameters(), lr=learning_rate)
                        else:
                            optimizer = Adadelta(gin.parameters())

                        '''config = {
                            'dropout': 0.5,  # 0.1, 0.5, 0.0
                            'hidden_units': [64, 64, 64, 64], [32, 32, 32, 32], [64], [32, 32]
                            "train_eps": True,  # try True, False
                            'aggregation': "sum",  # try 'mean', 'sum'
                        }'''

                        conf_count += 1
                        full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME, f"n_{conf_count}")
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

                        full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME, f"n_{conf_count}")
                        logger = Logger(filepath=os.path.join(full_experiment_path, "trainlog.txt"), mode="a")
                        if not USE_CLASS_WEIGHTS:
                            class_weights = None  # set class weights to None if not use class weights is selected
                        else:
                            class_weights = class_weights.to(
                                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                            )
                        logger.log(f"Launching training for experiment GIN with config: \n {config} with learning rate "
                                   f"{lr}, \n stored in "
                                   f"{full_experiment_path}...")

                        model, metrics = train_classifier(
                            gin,
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

    '''full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME, "best")
    logger = Logger(filepath=os.path.join(full_experiment_path, "trainlog.txt"), mode="a")
    constructor_params = model.serialize_constructor_params()
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(full_experiment_path, "state_dict.pt"))
    torch.save(constructor_params, os.path.join(full_experiment_path, "constructor_params.pt"))
    logger.log(f"Model trained and stored to {full_experiment_path}.")'''


if __name__ == '__main__':
    main()
