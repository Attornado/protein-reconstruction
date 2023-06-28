import math
import os
import random
import argparse
from typing import final
import torch
from torch.optim.lr_scheduler import LambdaLR
from log.logger import Logger
from torch_geometric.loader import ImbalancedSampler, DynamicBatchSampler
from models.classification.classifiers import MulticlassClassificationLoss
from models.classification.diffpool import DiffPool, DiffPoolMulticlassClassificationLoss
from models.classification.protmotionnet import PairedProtMotionNet, train_paired_classifier, \
    DiffPoolPairedProtMotionNet
from models.classification.sage import SAGEClassifier
from models.layers import GATConvBlock, GCNConvBlock, SAGEConvBlock
from models.pretraining.encoders import RevGCNEncoder, RevGATConvEncoder, RevSAGEConvEncoder, ResGCN2ConvEncoderV2
from models.pretraining.graph_infomax import DeepGraphInfomaxV2, MeanPoolReadout, RandomPermutationCorruption
from models.pretraining.gunet import GraphRevUNet, GraphUNetV2, HierarchicalTopKRevEncoder
from models.classification.ugformer import GCN, GAT, SAGE
from models.pretraining.normal_modes import EigenValueNMNet
from models.pretraining.vgae import VGAEv2, VGEncoder
from preprocessing.constants import PSCDB_CLEANED_TRAIN, PSCDB_CLEANED_VAL, DATA_PATH, \
    PSCDB_CLASS_WEIGHTS, PSCDB_PAIRED_CLASS_WEIGHTS, PSCDB_PAIRED_CLEANED_TRAIN, PSCDB_PAIRED_CLEANED_VAL, \
    PSCDB_PAIRED_CLEANED_TEST, RANDOM_SEED
from preprocessing.dataset.dataset_creation import load_dataset
from torch.optim import Adam, Adadelta
import torchinfo
from preprocessing.dataset.paired_dataset import PairedDataLoader
from training.training_tools import ACCURACY_METRIC, VAL_LOSS_METRIC, seed_everything


BATCH_SIZE: final = 10
EPOCHS: final = 1000
WARM_UP_EPOCHS: final = 80
WEIGHT_DECAY: final = 0
OPTIMIZER: final = "adam"
EARLY_STOPPING_PATIENCE: final = 35
EXPERIMENT_NAME: final = 'paired_protmotionnet_test6'
EXPERIMENT_PATH: final = os.path.join(DATA_PATH, "fitted", "classification", "pretrained_paired_protmotionnet")
PRE_TRAINED_MODEL_PATH: final = os.path.join(DATA_PATH, "fitted", "pretraining", "dgi", "dgi_gunet_gat_test0")
RESTORE_CHECKPOINT: final = True
USE_CLASS_WEIGHTS: final = True
USE_UNBALANCED_SAMPLER: final = False
USE_DYNAMIC_BATCH: final = True
DYNAMIC_BATCH_SIZE: final = 24000
LABEL_SMOOTHING: final = 0.1
IN_CHANNELS: final = 10
CONF_COUNT_START: final = 0
MODEL_NAME: final = "gunet"  # had GCN, GAT, SAGE, "diff_pool", "gunet", "grunet", "sage_c", "hier_rev"
MODEL_NAMES: final = frozenset([RevGATConvEncoder.MODEL_TYPE, RevSAGEConvEncoder.MODEL_TYPE, RevGCNEncoder.MODEL_TYPE,
                                "diff_pool", "gunet", "grunet", "sage_c", "hier_rev"])
PRE_TRAINED_MODEL_NAME: final = "dgi"
PRE_TRAINED_MODEL_NAMES: final = frozenset(["vgae", "dgi", "normal_mode"])
MONITORED_METRIC: final = VAL_LOSS_METRIC


def main(args):
    seed_everything(args.seed)
    ds_train = load_dataset(PSCDB_PAIRED_CLEANED_TRAIN, dataset_type="pscdb_paired")
    ds_val = load_dataset(PSCDB_PAIRED_CLEANED_VAL, dataset_type="pscdb_paired")
    # ds_test = load_dataset(PSCDB_CLEANED_TEST, dataset_type="pscdb")

    ys_train = torch.stack([data.y for data in ds_train], dim=0).view(-1)
    ys_val = torch.stack([data.y for data in ds_val], dim=0).view(-1)

    if args.use_unbalanced_sampler:
        sampler = ImbalancedSampler(dataset=ys_train, num_samples=int(len(ds_train)*1.5))
        sampler2 = ImbalancedSampler(dataset=ys_val, num_samples=int(len(ds_val)*3))
        dl_train = PairedDataLoader(ds_train, batch_size=args.batch_size, shuffle=False, sampler=sampler)
        dl_val = PairedDataLoader(ds_val, batch_size=args.batch_size, shuffle=False, sampler=sampler2)
    elif args.use_dynamic_batch:
        sampler = DynamicBatchSampler(ds_train, max_num=args.dynamic_batch_size)
        sampler2 = DynamicBatchSampler(ds_val, max_num=args.dynamic_batch_size)
        dl_train = PairedDataLoader(ds_train, batch_sampler=sampler, shuffle=False)
        dl_val = PairedDataLoader(ds_val, batch_sampler=sampler2, shuffle=False)
    else:
        dl_train = PairedDataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
        dl_val = PairedDataLoader(ds_val, batch_size=args.batch_size, shuffle=True)
    # dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)

    class_weights = torch.load(PSCDB_PAIRED_CLASS_WEIGHTS)
    in_channels = args.in_channels
    n_classes = len(class_weights)
    optim = args.optimizer

    try:
        path = os.path.join(args.experiment_path, args.experiment_name, "best_acc.pt")
        best_model_acc = torch.load(path)["best_acc"]
    except FileNotFoundError:
        best_model_acc = -1

    print(f"Loaded best_model_acc {best_model_acc}")
    conf_count = 0

    grid_values = {
        'dropout': [0.0, 0.3, 0.5],
        "dense_num": [1, 2],  # [2, 3]
        "learning_rate": [0.001, 0.0001, 0.00001, 0.000001]  # [0.0001, 0.00001, 0.000001]
    }

    m = args.model_name  # model type
    pretrained_model_type = args.pretrained_model_name
    print(args.pretrained_model_name)
    print(m)

    # Load params and weights
    constructor_params = torch.load(os.path.join(args.pretrained_model_path, "constructor_params.pt"))
    try:
        state_dict = torch.load(os.path.join(args.pretrained_model_path, "state_dict.pt"))
    except FileNotFoundError:
        state_dict = torch.load(os.path.join(args.pretrained_model_path, "state_dict.pt"))
    print(pretrained_model_type)
    # VGAE
    emb = None
    encoder = None
    if pretrained_model_type == "vgae":
        constructor_params_copy = dict(**constructor_params)
        del constructor_params_copy["encoder"]["state_dict"]
        print(f"Loaded constructor params: {constructor_params_copy}")
        del constructor_params_copy
        vgae = None
        if m == "sage_c":
            vgae = VGAEv2.from_constructor_params(
                constructor_params=constructor_params,
                vgencoder_constructor=VGEncoder,
                encoder_mu_constructor=SAGEClassifier
            )
            emb = vgae.encoder._encoder_mu.config_dict["dim_embedding"]*vgae.encoder._encoder_mu.config_dict["num_layers"]
            # nl = vgae.encoder._encoder_mu.config["num_layers"]
        elif m == RevGATConvEncoder.MODEL_TYPE:
            vgae = VGAEv2.from_constructor_params(
                constructor_params=constructor_params,
                vgencoder_constructor=VGEncoder,
                shared_encoder_constructor=RevGATConvEncoder,
                encoder_mu_constructor=GATConvBlock
            )
            emb = vgae.encoder._shared_encoder.out_channels
        elif m == RevGCNEncoder.MODEL_TYPE:
            vgae = VGAEv2.from_constructor_params(
                constructor_params=constructor_params,
                vgencoder_constructor=VGEncoder,
                shared_encoder_constructor=RevGCNEncoder,
                encoder_mu_constructor=GCNConvBlock
            )
            emb = vgae.encoder._shared_encoder.out_channels
        elif m == RevSAGEConvEncoder.MODEL_TYPE:
            vgae = VGAEv2.from_constructor_params(
                constructor_params=constructor_params,
                vgencoder_constructor=VGEncoder,
                shared_encoder_constructor=RevSAGEConvEncoder,
                encoder_mu_constructor=SAGEConvBlock
            )
            emb = vgae.encoder._shared_encoder.out_channels
        vgae.load_state_dict(state_dict)
        encoder = vgae.encoder
        encoder.standalone = True  # return only encoder mu

    # DGI
    if pretrained_model_type == "dgi":
        constructor_params_copy = dict(**constructor_params)
        del constructor_params_copy["encoder"]["state_dict"]
        print(f"Loaded constructor params: {constructor_params_copy}")
        del constructor_params_copy

        dgi = None
        if m == "sage_c":
            dgi = DeepGraphInfomaxV2.from_constructor_params(
                constructor_params=constructor_params,
                encoder_constructor=SAGEClassifier,
                readout=MeanPoolReadout,
                corruption=RandomPermutationCorruption(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            )
            emb = dgi.encoder.config["dim_embedding"]*dgi.encoder.config["num_layers"]
            nl = dgi.encoder.config["num_layers"]
        elif m == RevGATConvEncoder.MODEL_TYPE:
            dgi = DeepGraphInfomaxV2.from_constructor_params(
                constructor_params=constructor_params,
                encoder_constructor=RevGATConvEncoder,
                readout=MeanPoolReadout,
                corruption=RandomPermutationCorruption(
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                ),
            )
            emb = dgi.encoder.out_channels
        elif m == RevGCNEncoder.MODEL_TYPE:
            dgi = DeepGraphInfomaxV2.from_constructor_params(
                constructor_params=constructor_params,
                encoder_constructor=RevGCNEncoder,
                readout=MeanPoolReadout,
                corruption=RandomPermutationCorruption(
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                ),
            )
            emb = dgi.encoder.out_channels
        elif m == RevSAGEConvEncoder.MODEL_TYPE:
            dgi = DeepGraphInfomaxV2.from_constructor_params(
                constructor_params=constructor_params,
                encoder_constructor=RevSAGEConvEncoder,
                readout=MeanPoolReadout,
                corruption=RandomPermutationCorruption(
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                ),
            )
            emb = dgi.encoder.out_channels
        elif m == "gunet":
            dgi = DeepGraphInfomaxV2.from_constructor_params(
                constructor_params=constructor_params,
                encoder_constructor=GraphUNetV2,
                readout=MeanPoolReadout,
                corruption=RandomPermutationCorruption(
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                ),
            )
            emb = dgi.encoder.out_channels
        dgi.load_state_dict(state_dict)
        encoder = dgi.encoder

    evnmnet = None
    if pretrained_model_type == "normal_mode":
        print("Loaded constructor params: ", constructor_params)
        evnmnet = EigenValueNMNet.from_constructor_params(constructor_params=constructor_params)
        evnmnet.load_state_dict(state_dict)
        encoder = evnmnet._encoder

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
                        dense_units = [n_classes]
                        dense_activations = ["linear"]
                    elif dn == 2:
                        dense_units = [emb, n_classes]
                        dense_activations = ["gelu", "linear"]
                    elif dn == 3:
                        dense_units = [emb, int(emb/2), n_classes]
                        dense_activations = ["gelu", "gelu", "linear"]

                    if m == "diff_pool" and pretrained_model_type == "normal_mode":
                        diff_pool_config = evnmnet.encoder_params
                        model = DiffPoolPairedProtMotionNet(
                            diff_pool_config=diff_pool_config,
                            encoder_out_channels=emb,
                            dense_units=dense_units,
                            dense_activations=dense_activations,
                            dim_features=in_channels,
                            dropout=d
                        )
                        model._encoder.load_state_dict(encoder.state_dict())  # load pre-trained DiffPool weights
                    else:
                        encoder_out_channels = emb
                        forward_batch_index = False
                        if m == "grunet" or m == "gunet" or m == "sage_c" or m == "hier_rev":
                            forward_batch_index = True
                        model = PairedProtMotionNet(
                            encoder=encoder,
                            encoder_out_channels=encoder_out_channels,
                            dense_units=dense_units,
                            dense_activations=dense_activations,
                            dim_features=in_channels,
                            dropout=d,
                            readout=random.choice(["max_pool"]),
                            num_heads=2,  # try 2
                            forward_batch_index=forward_batch_index,
                            use_ff=True
                        )

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
                            lambda cur_epoch: (cur_epoch + 1) / args.warm_up_epochs \
                            if cur_epoch < args.warm_up_epochs else \
                            (0.5 * (1.0 + math.cos(math.pi * (cur_epoch - args.warm_up_epochs) /
                                                   (args.epochs - args.warm_up_epochs))))
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
                    if not args.use_class_weights:
                        class_weights = None
                        # set class weights to None if not use class weights is selected
                    else:
                        class_weights = class_weights.to(
                            torch.device("cuda") if torch.cuda.is_available() else
                            torch.device("cpu")
                        )
                    loss_fn = MulticlassClassificationLoss(weights=class_weights,
                                                           label_smoothing=args.label_smoothing)
                    if m == "diff_pool":
                        loss_fn = DiffPoolMulticlassClassificationLoss(weights=class_weights,
                                                                       label_smoothing=args.label_smoothing)
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
                        experiment_name=os.path.join(args.experiment_name, f"n_{conf_count}"),
                        early_stopping_patience=args.patience,
                        criterion=loss_fn,
                        logger=logger,
                        scheduler=scheduler,
                        monitor_metric=args.monitor_metric
                    )

                    if best_model_acc < metrics['accuracy']:
                        full_experiment_path = os.path.join(args.experiment_path, args.experiment_name)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='the integer value seed for global random state in Lightning')

    # Model's arguments
    parser.add_argument('--model_name', type=str, default=MODEL_NAME,
                        help=f"the model type (must be one of {MODEL_NAMES})")
    parser.add_argument('--pretrained_model_name', type=str, default=PRE_TRAINED_MODEL_NAME,
                        help=f"the pretrained model name (must be one of {PRE_TRAINED_MODEL_NAMES})")
    parser.add_argument('--in_channels', type=int, default=IN_CHANNELS, help="model input channels")
    parser.add_argument('--conf_count_start', type=int, default=CONF_COUNT_START,
                        help="the start grid search configuration")
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER,
                        help="the optimizer to use (either 'adam', 'adamw' or 'adadelta'")
    parser.add_argument('--warmup_steps', type=int, default=WARM_UP_EPOCHS,
                        help="the warmup epochs if using learning rate scheduler (AdamW optimizer)")
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY,
                        help="the weight decay rate or L2 regularization term (AdamW/Adam optimizer)")
    parser.add_argument('--label_smoothing', type=float, default=LABEL_SMOOTHING, help="the label smoothing term")
    parser.add_argument('--use_class_weights', type=bool, default=USE_CLASS_WEIGHTS,
                        help="whether to use class weights to handle unbalanced classes")

    # Training and checkpointing arguments
    parser.add_argument('--epochs', type=int, default=EPOCHS, help="the maximum number of epochs")
    parser.add_argument('--restore_checkpoint', type=bool, default=RESTORE_CHECKPOINT,
                        help="whether to restore old checkpoint to resume training")
    parser.add_argument('--pretrained_model_path', type=str, default=PRE_TRAINED_MODEL_PATH,
                        help='path to pre-trained model')
    parser.add_argument('--experiment_path', type=str, default=EXPERIMENT_PATH,
                        help='directory to save the experiments')
    parser.add_argument('--experiment_name', type=str, default=EXPERIMENT_NAME, help='experiment name')
    parser.add_argument('--monitor_metric', type=str, default=VAL_LOSS_METRIC,
                        help=f'metric to monitor for early stopping and checkpointing (either {VAL_LOSS_METRIC} or'
                             f' {ACCURACY_METRIC}')

    # Datamodule's arguments
    parser.add_argument('--use_dynamic_batch', type=bool, default=USE_DYNAMIC_BATCH,
                        help='whether to use dynamic batching')
    parser.add_argument('--use_unbalanced_sampler', type=bool, default=USE_UNBALANCED_SAMPLER,
                        help='whether to use unbalanced sampling in the data loader (no dynamic batch supported)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='size of the batch (if using static batch)')
    parser.add_argument('--dynamic_batch_size', type=int, default=DYNAMIC_BATCH_SIZE,
                        help='size of the batch (if using dynamic batch)')

    # Early stopping arguments
    parser.add_argument('--patience', type=int, default=EARLY_STOPPING_PATIENCE,
                        help='number of checks with no improvement after which training will be stopped')

    arguments = parser.parse_args()

    main(args=arguments)
