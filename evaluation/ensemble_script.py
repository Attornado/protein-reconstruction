import random
from typing import final
import torch
import json
from evaluation.constants import ENSEMBLE_CONFIG
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score, \
    top_k_accuracy_score
from sklearn.metrics import f1_score as f1_score_sklearn
from torchmetrics.functional import f1_score, auroc, accuracy, precision, recall
from torch_geometric.data import Data
from torch_geometric.loader import DynamicBatchSampler, DataLoader
from models.classification.protmotionnet import DiffPoolPairedProtMotionNet, PairedProtMotionNet
from models.ensemble import EnsembleGraphClassifier, VOTING, SOFTMAX_MEAN, load_classifier
from models.pretraining.gunet import GraphRevUNet, GraphUNetV2
from preprocessing.constants import RANDOM_SEED, PSCDB_PAIRED_CLEANED_VAL, PSCDB_CLEANED_VAL, PSCDB_CLEANED_TEST, \
    PSCDB_PAIRED_CLEANED_TEST
from preprocessing.dataset.dataset_creation import load_dataset
from preprocessing.dataset.paired_dataset import PairedDataLoader
from models.classification.sage import SAGEClassifier


IN_CHANNELS: final = 10
N_CLASSES: final = 7
ENSEMBLE_MODE: final = VOTING
PAIRED_CLASSIFIER: final = True
TOP_K: final = 1
BATCH_SIZE: final = 25
CLASS_DICTIONARY: final = {
    "diff_pool_protmotionnet": DiffPoolPairedProtMotionNet,
    "gunet_protmotionnet": PairedProtMotionNet,
    "grunet_protmotionnet": PairedProtMotionNet,
    "sage_c_protmotionnet": PairedProtMotionNet
}


@torch.no_grad()
def main():
    with open(ENSEMBLE_CONFIG, "r") as fp:
        ensemble_config = json.load(fp)
    checkpoints = ensemble_config["checkpoints"]
    constructor_dicts: list = ensemble_config["constructor_dicts"]
    classes: list = [cls_name for cls_name in ensemble_config["classes"]]
    weights: list = ensemble_config["weights"]

    models = []
    for checkpoint, constructor_dict, cls_name in zip(checkpoints, constructor_dicts, classes):
        print(f"Loading {constructor_dict}...")
        cls = CLASS_DICTIONARY[cls_name]
        if cls == PairedProtMotionNet:
            encoder_cls = None
            if cls_name == "grunet_protmotionnet":
                encoder_cls = GraphRevUNet
            elif cls_name == "gunet_protmotionnet":
                encoder_cls = GraphUNetV2
            elif cls_name == "sage_c_protmotionnet":
                encoder_cls = SAGEClassifier
            assert encoder_cls is not None
            model = load_classifier(path_checkpoint=checkpoint, path_config_dict=constructor_dict, cls=cls,
                                    encoder_constructor=encoder_cls)
        else:
            model = load_classifier(path_checkpoint=checkpoint, path_config_dict=constructor_dict, cls=cls,
                                    strict_weight_load=False)
        models.append(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_model = EnsembleGraphClassifier(models=models,
                                             dim_features=IN_CHANNELS,
                                             dim_target=N_CLASSES,
                                             ensemble_mode=SOFTMAX_MEAN,
                                             weights=weights,
                                             device=device)

    random.seed(RANDOM_SEED)
    if PAIRED_CLASSIFIER:
        ds_test = load_dataset(PSCDB_PAIRED_CLEANED_TEST, dataset_type="pscdb_paired")
        ds_val = load_dataset(PSCDB_PAIRED_CLEANED_VAL, dataset_type="pscdb_paired")
    else:
        ds_test = load_dataset(PSCDB_CLEANED_TEST, dataset_type="pscdb")
        ds_val = load_dataset(PSCDB_CLEANED_VAL, dataset_type="pscdb")

    if PAIRED_CLASSIFIER:
        dl_test = PairedDataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
        dl_val = PairedDataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
    else:
        dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    steps: int = 1
    running_accuracy: float = 0
    running_topk_acc: float = 0
    running_f1: float = 0
    running_precision: float = 0
    running_recall: float = 0
    running_auc: float = 0

    y_pred_all = []
    y_true_all = []
    probs_all = []

    for data in iter(dl_test):
        # move batch to device
        # data = data.to(device)

        if PAIRED_CLASSIFIER:
            before: Data = data.a
            after: Data = data.b
            x = before.x.float().to(device)
            edge_index = before.edge_index.to(device)
            batch_index = before.batch.to(device)
            x1 = after.x.float().to(device)
            edge_index1 = after.edge_index.to(device)
            batch_index1 = after.batch.to(device)
            probs = ensemble_model.ensemble(
                return_probs=True,
                x=x,
                edge_index=edge_index,
                batch_index=batch_index,
                x1=x1,
                edge_index1=edge_index1,
                batch_index1=batch_index1
            )
            y_pred = ensemble_model.ensemble(
                return_probs=False,
                x=x,
                edge_index=edge_index,
                batch_index=batch_index,
                x1=x1,
                edge_index1=edge_index1,
                batch_index1=batch_index1
            )
            del x, edge_index, edge_index1, batch_index, batch_index1, x1
        else:
            x = data.x.float().to(device)
            edge_index = data.edge_index.to(device)
            batch_index = data.batch.to(device)
            probs = ensemble_model.ensemble(
                return_probs=True,
                x=x,
                edge_index=edge_index,
                batch_index=batch_index
            )
            y_pred = ensemble_model.ensemble(
                return_probs=False,
                x=x,
                edge_index=edge_index,
                batch_index=batch_index
            )
            del x, edge_index, edge_index1, batch_index, batch_index1, x1

        y_pred_all.append(y_pred)
        probs_all.append(probs)
        y_true_all.append(data.y)

        torch.cuda.empty_cache()

        acc = accuracy(preds=probs, target=data.y.to(device), task='multiclass', num_classes=N_CLASSES, average="macro")
        if TOP_K is not None:
            top_k_acc = float(accuracy(preds=probs, target=data.y.to(device), task='multiclass', num_classes=N_CLASSES,
                                       top_k=TOP_K, average="macro"))
        else:
            top_k_acc = None
        prec = precision(preds=probs, target=data.a.y.to(device), task='multiclass', num_classes=N_CLASSES,
                         average="macro")
        rec = recall(preds=probs, target=data.a.y.to(device), task='multiclass', num_classes=N_CLASSES, average="macro")
        f1 = f1_score(preds=probs, target=data.a.y.to(device), task='multiclass', num_classes=N_CLASSES,
                      average="macro")
        # auc = roc_auc_score(y_true=data.y.cpu().numpy(), y_score=probs.cpu().detach().numpy(), multi_class="ovo",
        #                    average="macro")

        running_precision = running_precision + 1 / steps * (prec - running_precision)
        running_recall = running_recall + 1 / steps * (rec - running_recall)
        running_accuracy = running_accuracy + 1 / steps * (acc - running_accuracy)
        running_topk_acc = None if top_k_acc is None else running_topk_acc + 1 / steps * (top_k_acc - running_topk_acc)
        running_f1 = running_f1 + 1 / steps * (f1 - running_f1)
        # running_auc = running_auc + 1 / steps * (auc - running_auc)

        steps += 1
        print(f"Steps: {steps}/{len(ds_val) + 1}, running accuracy {running_accuracy}")

    print(f"Final metrics 1: accuracy: {running_accuracy}; precision: {running_precision}; recall: {running_recall}; "
          f"F1-score: {running_f1}; AUC: {running_auc}; top-{TOP_K} accuracy: {running_topk_acc}; ")

    y_true_all = torch.concat(y_true_all).detach().cpu().numpy()
    y_pred_all = torch.concat(y_pred_all).detach().cpu().numpy()
    probs_all = torch.concat(probs_all).detach().cpu().numpy()

    running_accuracy = accuracy_score(y_true_all, y_pred_all)
    running_recall = recall_score(y_true_all, y_pred_all, average="macro")
    running_precision = precision_score(y_true_all, y_pred_all, average="macro")
    running_f1 = f1_score_sklearn(y_true_all, y_pred_all, average="macro")
    running_auc = roc_auc_score(y_true_all, probs_all, average="macro", multi_class="ovo")
    balanced_acc = balanced_accuracy_score(y_true_all, y_pred_all, adjusted=False)
    #running_topk_acc = top_k_accuracy_score(y_true_all, y_pred_all, k=2, labels=[i for i in range(7)])

    print(f"Final metrics 1: accuracy: {running_accuracy}; precision: {running_precision}; recall: {running_recall}; "
          f"F1-score: {running_f1}; AUC: {running_auc}; top-{TOP_K} accuracy: {running_topk_acc}; Balanced acc: {balanced_acc}")


if __name__ == "__main__":
    main()
