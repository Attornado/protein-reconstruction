import os
from typing import final
import torch
from torch_geometric.loader import DataLoader
from log.logger import Logger
from models.pretraining.ugformer_unsup import UGformerV1, train_ugformer_unsup_inductive, compute_global_graph_indexes, \
    VOCAB_SIZE_KEY
from preprocessing.constants import PSCDB_CLEANED_TRAIN, PSCDB_CLEANED_VAL, DATA_PATH, \
    PSCDB_CLASS_WEIGHTS, PSCDB_GRAPH_INDEXES, PRETRAIN_CLEANED_TRAIN, PRETRAIN_GRAPH_INDEXES
from preprocessing.dataset.dataset_creation import load_dataset
from torch.optim import Adam, Adadelta
import torchinfo


BATCH_SIZE: final = 70  # was 200
MEDIUM_BATCH_SIZE: final = 50
SMALL_BATCH_SIZE: final = 25
TINY_BATCH_SIZE: final = 10
EPOCHS: final = 500
EARLY_STOPPING_PATIENCE: final = 25
EXPERIMENT_NAME: final = 'ugtransformer_test1'
EXPERIMENT_PATH: final = os.path.join(DATA_PATH, "fitted", "pretraining", "ugtransformer")
RESTORE_CHECKPOINT: final = True
USE_CLASS_WEIGHTS: final = True
LABEL_SMOOTHING: final = 0.0
IN_CHANNELS: final = 10
CONF_COUNT_START: final = 0
TRAIN_GRAPH_INDEXES: final = PRETRAIN_GRAPH_INDEXES
TRAIN_DATASET: final = PRETRAIN_CLEANED_TRAIN
TRAIN_DATASET_TYPE: final = "pretrain"
VAL_DATASET: final = PSCDB_CLEANED_VAL
VAL_DATASET_TYPE: final = "pscdb"
ADDITIONAL_TRAIN_DATASET: final = PSCDB_CLEANED_TRAIN
ADDITIONAL_TRAIN_DATASET_DATASET_TYPE: final = "pscdb"


def main():
    # ds_test = load_dataset(PSCDB_CLEANED_TEST, dataset_type="pscdb")
    if ADDITIONAL_TRAIN_DATASET is not None:
        additional_ds_train = load_dataset(ADDITIONAL_TRAIN_DATASET, dataset_type=ADDITIONAL_TRAIN_DATASET_DATASET_TYPE)
    else:
        additional_ds_train = None
    ds_train = load_dataset(TRAIN_DATASET, dataset_type=TRAIN_DATASET_TYPE)
    ds_val = load_dataset(VAL_DATASET, dataset_type=VAL_DATASET_TYPE)

    if ADDITIONAL_TRAIN_DATASET is not None:
        additional_dl_train = DataLoader(ds_train, batch_size=min(BATCH_SIZE, len(additional_ds_train)), shuffle=True)
    else:
        additional_dl_train = None

    dl_train = DataLoader(ds_train, batch_size=min(BATCH_SIZE, len(ds_train)), shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=min(BATCH_SIZE, len(ds_val)), shuffle=True)
    # dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)

    class_weights = torch.load(PSCDB_CLASS_WEIGHTS)
    in_channels = IN_CHANNELS
    n_classes = len(class_weights)
    l2 = 0.0  # try 5e-4
    optim = "adam"

    try:
        path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME, "best_auc.pt")
        best_model_auc = torch.load(path)["best_auc"]
    except FileNotFoundError:
        best_model_auc = -1

    # best_model_auc = 0.24839743971824646  # was -1
    print(f"Loaded best_model_auc {best_model_auc}")
    conf_count = 0

    # Load vocab size and global graph indexes
    try:
        global_graph_indexes = torch.load(TRAIN_GRAPH_INDEXES)
    except FileNotFoundError:
        global_graph_indexes = compute_global_graph_indexes(
            train_data=DataLoader(ds_train, batch_size=1, shuffle=False),
            path=TRAIN_GRAPH_INDEXES,
            use_tqdm=True
        )
    vocab_size = global_graph_indexes[VOCAB_SIZE_KEY]
    print(f"Loaded/created global graph indexes with vocab size {vocab_size}.")

    grid_values = {
        'dropout': [0.5],
        'embedding_dim': [None],
        'n_head': [1],
        "n_layers": [1, 2],
        "n_self_att_layers": [3, 4],
        'hidden_size': [512, 1024],
        'n_neighbours': [4, 8],
        "learning_rate": [5e-5, 5e-4]
    }

    for d in grid_values['dropout']:
        for emb in grid_values['embedding_dim']:
            for n in grid_values['n_head']:
                for nl in grid_values['n_layers']:
                    for nal in grid_values['n_self_att_layers']:
                        for h in grid_values['hidden_size']:
                            for ng in grid_values['n_neighbours']:
                                for lr in grid_values['learning_rate']:
                                    config = {
                                        'dropout': d,
                                        'hidden_size': h,
                                        'n_head': n,
                                        'n_neighbours': ng,
                                        'embedding_dim': emb,
                                        "n_self_att_layers": nal,
                                        "n_layers": nl,
                                        "learning_rate": lr
                                    }
                                    if conf_count < CONF_COUNT_START:
                                        conf_count += 1
                                        print(config)

                                    else:
                                        if (h == 1024 and (nal > 1 or nl > 1)) or (h == 512 and nal >= 3 or nl >= 2) \
                                                or (h == 1024 and ng >= 16):
                                            if h == 1024 and (nal >= 3 and nl >= 2) or (ng == 16 and nl >= 2):
                                                batch_size = TINY_BATCH_SIZE
                                            elif h == 1024 and nal >= 2 or nl >= 2:
                                                batch_size = SMALL_BATCH_SIZE
                                            else:
                                                batch_size = MEDIUM_BATCH_SIZE
                                            print(f"Chosen batch size: {batch_size}")
                                            dl_train = DataLoader(ds_train,
                                                                  batch_size=min(batch_size, len(ds_train)),
                                                                  shuffle=True)
                                            dl_val = DataLoader(ds_val,
                                                                batch_size=min(batch_size, len(ds_val)),
                                                                shuffle=True)
                                        else:
                                            print(f"Chosen batch size: {BATCH_SIZE}")
                                            dl_train = DataLoader(ds_train,
                                                                  batch_size=min(BATCH_SIZE, len(ds_train)),
                                                                  shuffle=True)
                                            dl_val = DataLoader(ds_val,
                                                                batch_size=min(BATCH_SIZE, len(ds_val)),
                                                                shuffle=True)
                                        learning_rate = lr
                                        ugformerv1 = UGformerV1(
                                            vocab_size=vocab_size,
                                            feature_dim_size=in_channels,
                                            ff_hidden_size=h,
                                            sampled_num=ng,
                                            num_self_att_layers=nal,
                                            num_gnn_layers=nl,
                                            embed_dim=emb,
                                            n_heads=n,
                                            dropout=d
                                        )

                                        if l2 > 0:
                                            optimizer = Adam(ugformerv1.parameters(), lr=learning_rate,
                                                             weight_decay=l2)
                                        elif optim == "adam":
                                            optimizer = Adam(ugformerv1.parameters(), lr=learning_rate)
                                        else:
                                            optimizer = Adadelta(ugformerv1.parameters())

                                        conf_count += 1
                                        full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME,
                                                                            f"n_{conf_count}")
                                        checkpoint_path = os.path.join(full_experiment_path, "checkpoint.pt")
                                        full_state_dict_path = os.path.join(full_experiment_path, "state_dict.pt")
                                        if RESTORE_CHECKPOINT and os.path.exists(checkpoint_path):
                                            print("Checkpoint found, loading state dict from checkpoint...")
                                            state_dict = torch.load(checkpoint_path)
                                            ugformerv1.load_state_dict(state_dict)
                                            print("State dict loaded.")
                                        elif RESTORE_CHECKPOINT and os.path.exists(full_state_dict_path):
                                            print("Final state dict found, loading state dict...")
                                            state_dict = torch.load(full_state_dict_path)
                                            ugformerv1.load_state_dict(state_dict)
                                            print("State dict loaded.")

                                        print(ugformerv1)
                                        print(torchinfo.summary(ugformerv1, depth=5))

                                        full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME,
                                                                            f"n_{conf_count}")
                                        logger = Logger(filepath=os.path.join(full_experiment_path, "trainlog.txt"),
                                                        mode="a")
                                        logger.log(f"Launching training for experiment UGFormerV1Unsup n{conf_count} "
                                                   f"with config \n {config} with learning rate "
                                                   f"{lr}, \n stored in "
                                                   f"{full_experiment_path}...")

                                        model, metrics = train_ugformer_unsup_inductive(
                                            ugformerv1,
                                            train_data=dl_train,
                                            val_data=dl_val,
                                            n_neighbours=ng,
                                            global_graph_indexes=global_graph_indexes,
                                            val_train_data=additional_dl_train,
                                            epochs=EPOCHS,
                                            optimizer=optimizer,
                                            experiment_path=EXPERIMENT_PATH,
                                            experiment_name=os.path.join(EXPERIMENT_NAME, f"n_{conf_count}"),
                                            early_stopping_patience=EARLY_STOPPING_PATIENCE,
                                            logger=logger
                                        )

                                        if best_model_auc < metrics['auc']:
                                            full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME)
                                            logger = Logger(filepath=os.path.join(full_experiment_path,
                                                                                  "trainlog.txt"),
                                                            mode="a")
                                            logger.log(f"Found better model than {best_model_auc} auc, "
                                                       f"with {metrics['auc']} auc, saving it in best dir")
                                            best_model_auc = metrics['auc']
                                            constructor_params = model.serialize_constructor_params()
                                            state_dict = model.state_dict()
                                            torch.save(state_dict, os.path.join(full_experiment_path,
                                                                                "state_dict.pt"))
                                            torch.save(constructor_params, os.path.join(full_experiment_path,
                                                                                        "constructor_params.pt"))
                                            torch.save({"best_auc": best_model_auc},
                                                       os.path.join(full_experiment_path, "best_auc.pt"))
                                            logger.log(f"Model with lr {lr} and config {config} \n trained and "
                                                       f"stored to {full_experiment_path}.")
                                        del model
                                        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
