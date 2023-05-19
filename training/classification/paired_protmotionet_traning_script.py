import math
import os
import random
from typing import final
import torch
from torch.optim.lr_scheduler import LambdaLR
from log.logger import Logger
from torch_geometric.loader import ImbalancedSampler, DynamicBatchSampler
from torch_geometric.nn.models import GraphUNet
from models.classification.classifiers import MulticlassClassificationLoss
from models.classification.diffpool import DiffPool, DiffPoolMulticlassClassificationLoss
from models.classification.protmotionnet import PairedProtMotionNet, train_paired_classifier, \
    DiffPoolPairedProtMotionNet
from models.pretraining.encoders import RevGCNEncoder, RevGATConvEncoder, RevSAGEConvEncoder, ResGCN2ConvEncoderV2
from models.pretraining.gunet import GraphRevUNet, GraphUNetV2
from models.classification.ugformer import GCN, GAT, SAGE
from preprocessing.constants import PSCDB_CLEANED_TRAIN, PSCDB_CLEANED_VAL, DATA_PATH, \
    PSCDB_CLASS_WEIGHTS, PSCDB_PAIRED_CLASS_WEIGHTS, PSCDB_PAIRED_CLEANED_TRAIN, PSCDB_PAIRED_CLEANED_VAL, \
    PSCDB_PAIRED_CLEANED_TEST, RANDOM_SEED
from preprocessing.dataset.dataset_creation import load_dataset
from torch.optim import Adam, Adadelta
import torchinfo
from preprocessing.dataset.paired_dataset import PairedDataLoader
from training.training_tools import ACCURACY_METRIC, VAL_LOSS_METRIC


BATCH_SIZE: final = 10
EPOCHS: final = 1000
WARM_UP_EPOCHS: final = 80
WEIGHT_DECAY: final = 1e-6
OPTIMIZER: final = "adamw"
EARLY_STOPPING_PATIENCE: final = 35
EXPERIMENT_NAME: final = 'paired_protmotionnet_test8'
EXPERIMENT_PATH: final = os.path.join(DATA_PATH, "fitted", "classification", "paired_protmotionnet")
RESTORE_CHECKPOINT: final = True
USE_CLASS_WEIGHTS: final = True
USE_UNBALANCED_SAMPLER: final = False
USE_DYNAMIC_BATCH: final = True
DYNAMIC_BATCH_SIZE: final = 24000
LABEL_SMOOTHING: final = 0.1
IN_CHANNELS: final = 10
CONF_COUNT_START: final = 0


def main():
    random.seed(RANDOM_SEED)
    ds_train = load_dataset(PSCDB_PAIRED_CLEANED_TRAIN, dataset_type="pscdb_paired")
    ds_val = load_dataset(PSCDB_PAIRED_CLEANED_VAL, dataset_type="pscdb_paired")
    # ds_test = load_dataset(PSCDB_CLEANED_TEST, dataset_type="pscdb")

    ys_train = torch.stack([data.y for data in ds_train], dim=0).view(-1)
    ys_val = torch.stack([data.y for data in ds_val], dim=0).view(-1)

    if USE_UNBALANCED_SAMPLER:
        sampler = ImbalancedSampler(dataset=ys_train, num_samples=int(len(ds_train)*1.5))
        sampler2 = ImbalancedSampler(dataset=ys_val, num_samples=int(len(ds_val)*3))
        dl_train = PairedDataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler)
        dl_val = PairedDataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler2)
    elif USE_DYNAMIC_BATCH:
        sampler = DynamicBatchSampler(ds_train, max_num=DYNAMIC_BATCH_SIZE)
        sampler2 = DynamicBatchSampler(ds_val, max_num=DYNAMIC_BATCH_SIZE)
        dl_train = PairedDataLoader(ds_train, batch_sampler=sampler, shuffle=False)
        dl_val = PairedDataLoader(ds_val, batch_sampler=sampler2, shuffle=False)
    else:
        dl_train = PairedDataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
        dl_val = PairedDataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True)
    # dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)

    class_weights = torch.load(PSCDB_PAIRED_CLASS_WEIGHTS)
    in_channels = IN_CHANNELS
    n_classes = len(class_weights)
    optim = OPTIMIZER

    try:
        path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME, "best_acc.pt")
        best_model_acc = torch.load(path)["best_acc"]
    except FileNotFoundError:
        best_model_acc = -1

    print(f"Loaded best_model_acc {best_model_acc}")
    conf_count = 0

    grid_values = {
        'dropout': [0.3, 0.5],
        "model_name": ["gunet"],  # had GCN, GAT, SAGE, "diff_pool", "gunet"
        'embedding_dim': [128],  # [32, 64, 128, 256]
        'n_heads_gat': [8],  # [8]
        "dense_num": [1, 2],  # [2, 3]
        "n_layers": [4],  # [1, 5, 20, 50, 100]
        "learning_rate": [0.0001, 0.00001]  # [0.0001, 0.00001, 0.000001]
    }

    for m in grid_values['model_name']:
        for nl in grid_values['n_layers']:
            for emb in grid_values['embedding_dim']:
                for dn in grid_values['dense_num']:
                    for d in grid_values['dropout']:
                        for lr in grid_values['learning_rate']:
                            for nh in grid_values['n_heads_gat'] if m == GAT else [1]:
                                d = random.choice([d, d, d, 0.5, 0.4])
                                if nl == 50:
                                    nl = random.choice([nl, 80])
                                nh = random.choice([nh, 16])
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
                                    dense_units = None
                                    dense_activations = None
                                    if dn == 1:
                                        dense_units = [n_classes]
                                        dense_activations = ["linear"]
                                    elif dn == 2:
                                        dense_units = [emb, n_classes]
                                        dense_activations = ["gelu", "linear"]
                                    elif dn == 3:
                                        dense_units = [emb, int(emb/2), n_classes]
                                        dense_activations = ["gelu", "gelu", "linear"]

                                    encoder = None
                                    if m == RevGATConvEncoder.MODEL_TYPE:
                                        encoder = RevGATConvEncoder(
                                            in_channels=in_channels,
                                            hidden_channels=emb,
                                            out_channels=emb,
                                            heads=nh,
                                            num_convs=nl,
                                            dropout=d,
                                            num_groups=2
                                        )
                                    elif m == RevSAGEConvEncoder.MODEL_TYPE:
                                        encoder = RevSAGEConvEncoder(
                                            in_channels=in_channels,
                                            hidden_channels=emb,
                                            out_channels=emb,
                                            num_convs=nl,
                                            dropout=d,
                                            num_groups=2
                                        )
                                    elif m == RevGCNEncoder.MODEL_TYPE:
                                        encoder = RevGCNEncoder(
                                            in_channels=in_channels,
                                            hidden_channels=emb,
                                            out_channels=emb,
                                            num_convs=nl,
                                            dropout=d,
                                            improved=random.choice([True, False]),
                                            num_groups=2
                                        )
                                    elif m == "grunet":
                                        encoder = GraphRevUNet(
                                            in_channels=in_channels,
                                            hidden_channels=emb,
                                            out_channels=emb,
                                            num_convs=[nl, nl, nl, nl],
                                            dropout=d,
                                            pool_ratio=0.8,
                                            model_type=RevGCNEncoder.MODEL_TYPE,
                                            num_groups=2
                                        )
                                    elif m == "gunet":
                                        if nl == 3:
                                            pool_ratios = [0.9, 0.7, 0.6]
                                        elif nl == 4:
                                            pool_ratios = [0.9, 0.7, 0.6, 0.5]
                                        elif nl == 5:
                                            pool_ratios = [0.9, 0.8, 0.7, 0.6, 0.5]
                                        else:
                                            pool_ratios = 0.7
                                        encoder = GraphUNetV2(
                                            in_channels=in_channels,
                                            hidden_channels=emb,
                                            out_channels=emb,
                                            depth=nl,
                                            pool_ratios=pool_ratios,
                                            sum_res=True,
                                            act="relu"
                                        )

                                    if m == "diff_pool":
                                        model = DiffPoolPairedProtMotionNet(
                                            diff_pool_config={'num_layers': nl, 'dim_embedding': 256,
                                                              'gnn_dim_hidden': 128, 'dim_embedding_MLP': emb,
                                                              'max_num_nodes': 3787},
                                            encoder_out_channels=emb,
                                            dense_units=dense_units,
                                            dense_activations=dense_activations,
                                            dim_features=in_channels,
                                            dropout=d
                                        )
                                    else:
                                        encoder_out_channels = emb*nl if m == "gunet" and not encoder.sum_res else emb
                                        model = PairedProtMotionNet(
                                            encoder=encoder,
                                            encoder_out_channels=encoder_out_channels,
                                            dense_units=dense_units,
                                            dense_activations=dense_activations,
                                            dim_features=in_channels,
                                            dropout=d,
                                            readout=random.choice(["mean_pool"]),
                                            num_heads=4,  # try 2
                                            forward_batch_index=True if m == "grunet" or m == "gunet" else False,
                                            use_ff=True
                                        )

                                    l2 = WEIGHT_DECAY
                                    scheduler = None
                                    if l2 > 0 and optim == "adam":
                                        optimizer = Adam(model.parameters(), lr=learning_rate,
                                                         weight_decay=l2)
                                    elif optim == "adam":
                                        optimizer = Adam(model.parameters(), lr=learning_rate)
                                    elif optim == "adamw":
                                        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                                                      betas=(0.9, 0.999), eps=1e-8,
                                                                      weight_decay=WEIGHT_DECAY)
                                        # warm_up + cosine weight decay
                                        lr_plan = \
                                            lambda cur_epoch: (cur_epoch + 1) / WARM_UP_EPOCHS \
                                            if cur_epoch < WARM_UP_EPOCHS else \
                                            (0.5 * (1.0 + math.cos(math.pi * (cur_epoch - WARM_UP_EPOCHS) /
                                                                   (EPOCHS - WARM_UP_EPOCHS))))
                                        scheduler = LambdaLR(optimizer, lr_lambda=lr_plan)
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
                                    loss_fn = MulticlassClassificationLoss(weights=class_weights,
                                                                           label_smoothing=LABEL_SMOOTHING)
                                    if m == "diff_pool":
                                        loss_fn = DiffPoolMulticlassClassificationLoss(weights=class_weights,
                                                                                       label_smoothing=LABEL_SMOOTHING)
                                        config = model.serialize_constructor_params()
                                    if m == "gunet" or m == "grunet":
                                        config = model.serialize_constructor_params()
                                        del config["encoder"]["state_dict"]
                                    logger.log(f"Launching training for experiment PairedProtMotionNet n{conf_count} "
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
                                        criterion=loss_fn,
                                        logger=logger,
                                        scheduler=scheduler,
                                        monitor_metric=VAL_LOSS_METRIC
                                    )

                                    if best_model_acc < metrics['accuracy']:
                                        full_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_NAME)
                                        logger = Logger(filepath=os.path.join(full_experiment_path, "trainlog.txt"),
                                                        mode="a")
                                        logger.log(f"Found better model n{conf_count} than {best_model_acc} acc, "
                                                   f"with accuracy "
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
    main()
