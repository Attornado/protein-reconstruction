import os
from typing import final
import torch
from torch_geometric.loader import DataLoader
from log.logger import Logger
from models.classification.classifiers import train_classifier, MulticlassClassificationLoss
from models.classification.ugformer import UGFormerV2, GCN, GAT, SAGE
from preprocessing.constants import PSCDB_CLEANED_TRAIN, PSCDB_CLEANED_VAL, DATA_PATH, \
    PSCDB_CLASS_WEIGHTS
from preprocessing.dataset.dataset_creation import load_dataset
from torch.optim import Adam, Adadelta
import torchinfo


BATCH_SIZE: final = 30
EPOCHS: final = 2000
EARLY_STOPPING_PATIENCE: final = 25
EXPERIMENT_NAME: final = 'ugtransformer_test0'
EXPERIMENT_PATH: final = os.path.join(DATA_PATH, "fitted", "classification", "ugtransformer")
RESTORE_CHECKPOINT: final = True
USE_CLASS_WEIGHTS: final = True
LABEL_SMOOTHING: final = 0.0
IN_CHANNELS: final = 10
CONF_COUNT_START: final = 2297


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
        'dropout': [0.1, 0.5],
        "model_name": [SAGE, GCN, GAT],  # had GCN_MODEL_TYPE, GAT_MODEL_TYPE
        'hidden_size': [32, 64, 128, 256],
        'n_head': [1, 2, 4],
        'n_heads_gat': [4],
        'embedding_dim': ["feature", "hidden"],
        "n_self_att_layers": [1, 2, 3],
        "n_layers": [1, 2, 3],
        "learning_rate": [0.0001, 0.000001, 0.0000001]
    }

    for d in grid_values['dropout']:
        for m in grid_values['model_name']:
            for h in grid_values['hidden_size']:
                for n in grid_values['n_head']:
                    for nl in grid_values['n_layers']:
                        for emb in grid_values['embedding_dim']:
                            for nal in grid_values['n_self_att_layers']:
                                for lr in grid_values['learning_rate']:

                                    if conf_count < CONF_COUNT_START or n == 4 or lr == 0.0001 or d == 0.1 or h == 64 \
                                            or nal == 3 or nl == 3 or m == SAGE or m == GCN:
                                        conf_count += 1
                                        config = {
                                            'dropout': d,
                                            "model_name": m,
                                            'hidden_size': h,  # try 10, 32, 64, 50, 100, 200
                                            'out_dim': h,  # try 10, 20, 50, 100, 200,
                                            'n_head': n,
                                            "n_self_att_layers": nal,
                                            "n_layers": nl,
                                            "embedding_dim": h if emb == 'hidden' else in_channels,
                                        }
                                        print(config)

                                    else:

                                        config = {
                                            'dropout': d,
                                            "model_name": m,
                                            'hidden_size': h,  # try 10, 32, 64, 50, 100, 200
                                            'out_dim': h,  # try 10, 20, 50, 100, 200,
                                            'n_head': n,
                                            "n_self_att_layers": nal,
                                            "n_layers": nl,
                                            "embedding_dim": h if emb == 'hidden' else in_channels,
                                        }
                                        learning_rate = lr
                                        if config['model_name'] == GAT:
                                            for gn in grid_values['n_heads_gat']:
                                                if conf_count < CONF_COUNT_START:
                                                    conf_count += 1

                                                else:
                                                    config['heads'] = gn
                                                    ugformerv2 = UGFormerV2(dim_features=in_channels,
                                                                            hidden_size=h,
                                                                            n_self_att_layers=nal,
                                                                            n_layers=nl,
                                                                            n_head=n,
                                                                            dropout=d,
                                                                            embedding_size=config['embedding_dim'],
                                                                            dim_target=n_classes,
                                                                            conv_type=m)

                                                    if l2 > 0:
                                                        optimizer = Adam(ugformerv2.parameters(), lr=learning_rate,
                                                                         weight_decay=l2)
                                                    elif optim == "adam":
                                                        optimizer = Adam(ugformerv2.parameters(), lr=learning_rate)
                                                    else:
                                                        optimizer = Adadelta(ugformerv2.parameters())

                                                    conf_count += 1
                                                    full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME,
                                                                                        f"n_{conf_count}")
                                                    checkpoint_path = os.path.join(full_experiment_path, "checkpoint.pt")
                                                    full_state_dict_path = os.path.join(full_experiment_path,
                                                                                        "state_dict.pt")
                                                    if RESTORE_CHECKPOINT and os.path.exists(checkpoint_path):
                                                        print("Checkpoint found, loading state dict from checkpoint...")
                                                        state_dict = torch.load(checkpoint_path)
                                                        ugformerv2.load_state_dict(state_dict)
                                                        print("State dict loaded.")
                                                    elif RESTORE_CHECKPOINT and os.path.exists(full_state_dict_path):
                                                        print("Final state dict found, loading state dict...")
                                                        state_dict = torch.load(full_state_dict_path)
                                                        ugformerv2.load_state_dict(state_dict)
                                                        print("State dict loaded.")

                                                    print(ugformerv2)
                                                    print(torchinfo.summary(ugformerv2, depth=5))

                                                    full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME,
                                                                                        f"n_{conf_count}")
                                                    logger = Logger(filepath=os.path.join(full_experiment_path,
                                                                                          "trainlog.txt"),
                                                                    mode="a")
                                                    if not USE_CLASS_WEIGHTS:
                                                        class_weights = None
                                                        # set class weights to None if not use class weights is selected
                                                    else:
                                                        class_weights = class_weights.to(
                                                            torch.device("cuda") if torch.cuda.is_available() else
                                                            torch.device("cpu")
                                                        )
                                                    logger.log(f"Launching training for experiment UGFormerV2 "
                                                               f"n{conf_count} with config: \n {config} with learning rate "
                                                               f"{lr}, \n stored in "
                                                               f"{full_experiment_path}...")

                                                    model, metrics = train_classifier(
                                                        ugformerv2,
                                                        train_data=dl_train,
                                                        val_data=dl_val,
                                                        epochs=EPOCHS,
                                                        optimizer=optimizer,
                                                        experiment_path=EXPERIMENT_PATH,
                                                        experiment_name=os.path.join(EXPERIMENT_NAME, f"n_{conf_count}"),
                                                        early_stopping_patience=EARLY_STOPPING_PATIENCE,
                                                        criterion=MulticlassClassificationLoss(
                                                            weights=class_weights,
                                                            label_smoothing=LABEL_SMOOTHING
                                                        ),
                                                        logger=logger
                                                    )

                                                    if best_model_acc < metrics['accuracy']:
                                                        full_experiment_path = os.path.join(EXPERIMENT_PATH,
                                                                                            EXPERIMENT_NAME)
                                                        logger = Logger(filepath=os.path.join(full_experiment_path,
                                                                                              "trainlog.txt"),
                                                                        mode="a")
                                                        logger.log(f"Found better model than {best_model_acc} acc, with "
                                                                   f"accuracy {metrics['accuracy']} acc, saving it in best"
                                                                   f" dir")
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
                                                        logger.log(f"Model with lr {lr} and config {config} "
                                                                   f"\n trained and stored to {full_experiment_path}.")
                                                    del model
                                        else:
                                            learning_rate = lr
                                            ugformerv2 = UGFormerV2(dim_features=in_channels,
                                                                    hidden_size=h,
                                                                    n_self_att_layers=nal,
                                                                    n_layers=nl,
                                                                    n_head=n,
                                                                    dropout=d,
                                                                    embedding_size=h if emb == 'hidden'
                                                                    else in_channels,
                                                                    dim_target=n_classes,
                                                                    conv_type=m
                                                                    )

                                            if l2 > 0:
                                                optimizer = Adam(ugformerv2.parameters(), lr=learning_rate,
                                                                 weight_decay=l2)
                                            elif optim == "adam":
                                                optimizer = Adam(ugformerv2.parameters(), lr=learning_rate)
                                            else:
                                                optimizer = Adadelta(ugformerv2.parameters())

                                            conf_count += 1
                                            full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME,
                                                                                f"n_{conf_count}")
                                            checkpoint_path = os.path.join(full_experiment_path, "checkpoint.pt")
                                            full_state_dict_path = os.path.join(full_experiment_path, "state_dict.pt")
                                            if RESTORE_CHECKPOINT and os.path.exists(checkpoint_path):
                                                print("Checkpoint found, loading state dict from checkpoint...")
                                                state_dict = torch.load(checkpoint_path)
                                                ugformerv2.load_state_dict(state_dict)
                                                print("State dict loaded.")
                                            elif RESTORE_CHECKPOINT and os.path.exists(full_state_dict_path):
                                                print("Final state dict found, loading state dict...")
                                                state_dict = torch.load(full_state_dict_path)
                                                ugformerv2.load_state_dict(state_dict)
                                                print("State dict loaded.")

                                            print(ugformerv2)
                                            print(torchinfo.summary(ugformerv2, depth=5))

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

                                            model, metrics = train_classifier(
                                                ugformerv2,
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
                                                logger = Logger(filepath=os.path.join(full_experiment_path,
                                                                                      "trainlog.txt"),
                                                                mode="a")
                                                logger.log(f"Found better model than {best_model_acc} acc, "
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

    '''
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
    '''


if __name__ == '__main__':
    main()
