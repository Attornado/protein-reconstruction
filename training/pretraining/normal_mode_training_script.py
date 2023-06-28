import math
import os
import argparse
from typing import final
import torch
from torch.nn import MSELoss
from torch.optim.lr_scheduler import LambdaLR
from log.logger import Logger
from torch_geometric.loader import DynamicBatchSampler, DataLoader
from models.classification.sage import SAGEClassifier
from models.layers import GATConvBlock, GCNConvBlock, SAGEConvBlock
from models.pretraining.encoders import RevGCNEncoder, RevGATConvEncoder, RevSAGEConvEncoder
from models.pretraining.normal_modes import EigenValueNMNet, DIFF_POOL, SAGE, GUNET, train_nm_net, N_EIGENVALUES_DEFAULT
from preprocessing.constants import DATA_PATH, RANDOM_SEED, PRETRAIN_CLEANED_TRAIN, PRETRAIN_CLEANED_VAL
from preprocessing.dataset.dataset_creation import load_dataset
from torch.optim import Adam, Adadelta
import torchinfo
from training.training_tools import ACCURACY_METRIC, VAL_LOSS_METRIC, seed_everything


BATCH_SIZE: final = 10
EPOCHS: final = 100
WARM_UP_EPOCHS: final = 30
WEIGHT_DECAY: final = 0  # try 1e-5
OPTIMIZER: final = "adamw"
EARLY_STOPPING_PATIENCE: final = 15
EXPERIMENT_NAME: final = 'normal_mode_diff_pool_test0'
EXPERIMENT_PATH: final = os.path.join(DATA_PATH, "fitted", "pretraining", "normal_modes")
RESTORE_CHECKPOINT: final = True
USE_DYNAMIC_BATCH: final = True
DYNAMIC_BATCH_SIZE: final = 18000
IN_CHANNELS: final = 10
CONF_COUNT_START: final = 0
MODEL_NAME: final = "diff_pool"  # had GCN, GAT, SAGE, "diff_pool", "gunet", "grunet", "sage_c", "hier_rev"
MODEL_NAMES: final = frozenset([RevGATConvEncoder.MODEL_TYPE, RevSAGEConvEncoder.MODEL_TYPE, RevGCNEncoder.MODEL_TYPE,
                                "diff_pool", "gunet", "grunet", "sage_c", "hier_rev"])
MONITORED_METRIC: final = VAL_LOSS_METRIC


def main(args):
    seed_everything(args.seed)
    ds_train = load_dataset(PRETRAIN_CLEANED_TRAIN, dataset_type="pretrain")
    ds_val = load_dataset(PRETRAIN_CLEANED_VAL, dataset_type="pretrain")
    # ds_test = load_dataset(PSCDB_CLEANED_TEST, dataset_type="pscdb")

    # ys_train = torch.stack([data.y for data in ds_train], dim=0).view(-1)
    # ys_val = torch.stack([data.y for data in ds_val], dim=0).view(-1)

    if args.use_dynamic_batch:
        sampler = DynamicBatchSampler(ds_train, max_num=args.dynamic_batch_size)
        sampler2 = DynamicBatchSampler(ds_val, max_num=args.dynamic_batch_size)
        dl_train = DataLoader(ds_train, batch_sampler=sampler, shuffle=False)
        dl_val = DataLoader(ds_val, batch_sampler=sampler2, shuffle=False)
    else:
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=True)
    # dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)

    in_channels = args.in_channels
    optim = args.optimizer

    try:
        path = os.path.join(args.experiment_path, args.experiment_name, "best_mse.pt")
        best_model_acc = torch.load(path)["best_mse"]
    except FileNotFoundError:
        best_model_acc = -1

    print(f"Loaded best_model_acc {best_model_acc}")
    conf_count = 0

    grid_values = {
        'dropout': [0.1, 0.3, 0.5],
        "dense_num": [1, 2],  # [2, 3]
        "learning_rate": [0.001, 0.0001, 0.00001]  # [0.0001, 0.00001, 0.000001]
    }

    m = args.model_name  # model type
    emb = args.embedding_dim
    nl = args.num_layers

    for dn in grid_values['dense_num']:
        for d in grid_values['dropout']:
            for lr in grid_values['learning_rate']:

                config = {
                    'dropout': d,
                    "model_name": m,
                    "dense_num": dn,
                    'learning_rate': lr
                }

                if conf_count < args.conf_count_start:
                    conf_count += 1
                    print(config)

                else:
                    learning_rate = lr
                    dense_units = None
                    dense_activations = None
                    if dn == 1:
                        dense_units = []
                        dense_activations = []
                    elif dn == 2:
                        dense_units = [emb]
                        dense_activations = ["gelu"]
                    elif dn == 3:
                        dense_units = [emb, int(emb/2)]
                        dense_activations = ["gelu", "gelu"]

                    if m == "diff_pool":
                        model = EigenValueNMNet(
                            in_channels=in_channels,
                            encoder_out_channels=emb,
                            dense_units=dense_units,
                            dense_activations=dense_activations,
                            encoder_type=DIFF_POOL,
                            n_eigenvalues=N_EIGENVALUES_DEFAULT,
                            dropout=d,
                            **{'num_layers': nl,
                               'dim_embedding': 256,
                               'gnn_dim_hidden': 128,
                               'dim_embedding_MLP': emb,
                               'max_num_nodes': 3787}
                        )
                        # Model with lr 1e-05 and config {'encoder_out_channels': 200, 'dense_units': [100, 7],
                        # 'dense_activations': ['gelu', 'linear'], 'dropout': 0.1, 'forward_batch_index': False,
                        # 'dim_features': 10, 'use_ff': False, 'diff_pool_config': {'num_layers': 3,
                        # 'dim_embedding': 256, 'gnn_dim_hidden': 128, 'dim_embedding_MLP': 100, 'max_num_nodes': 3787}}
                    elif m == "sage_c":
                        model = EigenValueNMNet(
                            in_channels=in_channels,
                            encoder_out_channels=emb,
                            dense_units=dense_units,
                            dense_activations=dense_activations,
                            encoder_type=SAGE,
                            dropout=d,  # should be 0.0/0.1
                            **{"num_layers": nl,  # should be 3/
                               "aggregation": "mean",  # can be "mean" or "max"
                               "dim_embedding": emb,  # should be 256
                               "return_embeddings": True}
                        )
                        # Model with lr 1e-05 and config {'dropout': 0.0, 'model_name': 'sage_c', 'n_layers': 3,
                        # 'embedding_dim': 256, 'dense_num': 2, 'learning_rate': 1e-05}
                    else:
                        if nl == 3:
                            pool_ratios = [0.9, 0.7, 0.6]
                        elif nl == 4:
                            pool_ratios = [0.9, 0.7, 0.6, 0.5]
                        elif nl == 5:
                            pool_ratios = [0.9, 0.8, 0.7, 0.6, 0.5]
                        else:
                            pool_ratios = 0.7
                        model = EigenValueNMNet(
                            in_channels=in_channels,
                            encoder_out_channels=emb,  # should be 128
                            dense_units=dense_units,
                            dense_activations=dense_activations,
                            encoder_type=GUNET,
                            dropout=d,  # should be 0.3/0.5
                            **{"depth": nl,  # should be 4
                               "pool_ratios": pool_ratios,
                               "sum_res": False}
                        )
                        # Model with lr 1e-05 and config {'encoder_out_channels': 128, 'dense_units': [7],
                        # 'dense_activations': ['linear'], 'dropout': 0.5, 'readout': 'max_pool',
                        # 'forward_batch_index': True, 'dim_features': 10, 'encoder': {
                        # 'constructor_params': {'in_channels': 10, 'hidden_channels': 128, 'out_channels': 128,
                        # 'depth': 4, 'pool_ratios': [0.9, 0.7, 0.6, 0.5], 'sum_res': False, 'act': 'relu'}},
                        # 'num_heads': 2, 'vdim': None, 'kdim': None, 'use_ff': True}

                    l2 = args.weight_decay
                    scheduler = None
                    if l2 > 0 and optim == "adam":
                        optimizer = Adam(model.parameters(), lr=learning_rate,
                                         weight_decay=l2)
                    elif optim == "adam":
                        optimizer = Adam(model.parameters(), lr=learning_rate)
                    elif optim == "adamw":
                        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                                      betas=(0.9, 0.999), eps=1e-8,
                                                      weight_decay=l2)
                        # warm_up + cosine weight decay
                        lr_plan = \
                            lambda cur_epoch: (cur_epoch + 1) / args.warm_up_steps \
                                if cur_epoch < args.warm_up_steps else \
                                (0.5 * (1.0 + math.cos(math.pi * (cur_epoch - args.warm_up_steps) /
                                                       (args.epochs - args.warm_up_steps))))
                        scheduler = LambdaLR(optimizer, lr_lambda=lr_plan)
                    else:
                        optimizer = Adadelta(model.parameters())

                    conf_count += 1
                    full_experiment_path = os.path.join(args.experiment_path, args.experiment_name,
                                                        f"n_{conf_count}")
                    checkpoint_path = os.path.join(full_experiment_path, "checkpoint.pt")
                    full_state_dict_path = os.path.join(full_experiment_path, "state_dict.pt")
                    if args.restore_checkpoint and os.path.exists(checkpoint_path):
                        print("Checkpoint found, loading state dict from checkpoint...")
                        state_dict = torch.load(checkpoint_path)
                        model.load_state_dict(state_dict)
                        print("State dict loaded.")
                    elif args.restore_checkpoint and os.path.exists(full_state_dict_path):
                        print("Final state dict found, loading state dict...")
                        state_dict = torch.load(full_state_dict_path)
                        model.load_state_dict(state_dict)
                        print("State dict loaded.")

                    print(model)
                    print(torchinfo.summary(model, depth=5))

                    full_experiment_path = os.path.join(args.experiment_path, args.experiment_name,
                                                        f"n_{conf_count}")
                    logger = Logger(filepath=os.path.join(full_experiment_path, "trainlog.txt"),
                                    mode="a")
                    if m == "diff_pool":
                        config = model.serialize_constructor_params()
                    if m == "gunet" or m == "sage_c":
                        config = model.serialize_constructor_params()
                        del config["encoder"]["state_dict"]
                    logger.log(f"Launching training for experiment EigenValueNMNet n{conf_count} "
                               f"with config \n {config} with learning rate "
                               f"{lr}, \n stored in "
                               f"{full_experiment_path}...")

                    model, metrics = train_nm_net(
                        model,
                        train_data=dl_train,
                        val_data=dl_val,
                        epochs=EPOCHS,
                        optimizer=optimizer,
                        experiment_path=EXPERIMENT_PATH,
                        experiment_name=os.path.join(args.experiment_name, f"n_{conf_count}"),
                        early_stopping_patience=args.patience,
                        criterion=MSELoss(),
                        logger=logger,
                        scheduler=scheduler,
                        monitored_metric=args.monitor_metric
                    )

                    if best_model_acc < metrics['accuracy']:
                        full_experiment_path = os.path.join(args.experiment_path, args.experiment_name)
                        logger = Logger(filepath=os.path.join(full_experiment_path, "trainlog.txt"),
                                        mode="a")
                        logger.log(f"Found better model n{conf_count} than {best_model_acc} acc, "
                                   f"with mse "
                                   f"{metrics['accuracy']} acc, saving it in best dir")
                        best_model_acc = metrics['accuracy']
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='the integer value seed for global random state in Lightning')

    # Model's arguments
    parser.add_argument('--model_name', type=str, default=MODEL_NAME,
                        help=f"the model type (must be one of {MODEL_NAMES})")
    parser.add_argument('--embedding_dim', type=int, default=100, help=f"the embedding size")
    parser.add_argument('--num_layers', type=int, default=5, help=f"the number of layers")
    parser.add_argument('--in_channels', type=int, default=IN_CHANNELS, help="model input channels")
    parser.add_argument('--conf_count_start', type=int, default=CONF_COUNT_START,
                        help="the start grid search configuration")
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER,
                        help="the optimizer to use (either 'adam', 'adamw' or 'adadelta'")
    parser.add_argument('--warm_up_steps', type=int, default=WARM_UP_EPOCHS,
                        help="the warmup epochs if using learning rate scheduler (AdamW optimizer)")
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY,
                        help="the weight decay rate or L2 regularization term (AdamW/Adam optimizer)")

    # Training and checkpointing arguments
    parser.add_argument('--epochs', type=int, default=EPOCHS, help="the maximum number of epochs")
    parser.add_argument('--restore_checkpoint', type=bool, default=RESTORE_CHECKPOINT,
                        help="whether to restore old checkpoint to resume training")
    parser.add_argument('--experiment_path', type=str, default=EXPERIMENT_PATH,
                        help='directory to save the experiments')
    parser.add_argument('--experiment_name', type=str, default=EXPERIMENT_NAME, help='experiment name')
    parser.add_argument('--monitor_metric', type=str, default=VAL_LOSS_METRIC,
                        help=f'metric to monitor for early stopping and checkpointing (either {VAL_LOSS_METRIC} or'
                             f' {ACCURACY_METRIC}')

    # Datamodule's arguments
    parser.add_argument('--use_dynamic_batch', type=bool, default=USE_DYNAMIC_BATCH,
                        help='whether to use dynamic batching')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='size of the batch (if using static batch)')
    parser.add_argument('--dynamic_batch_size', type=int, default=DYNAMIC_BATCH_SIZE,
                        help='size of the batch (if using dynamic batch)')

    # Early stopping arguments
    parser.add_argument('--patience', type=int, default=EARLY_STOPPING_PATIENCE,
                        help='number of checks with no improvement after which training will be stopped')

    arguments = parser.parse_args()

    main(args=arguments)
