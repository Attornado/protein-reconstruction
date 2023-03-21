import os
import shutil
from typing import final
import pandas as pd
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from preprocessing.constants import PSCDB_PATH, PSCDB_CLEANED_TRAIN, PRETRAIN_CLEANED_TRAIN, PATH_PDBS_JSON, \
    PRETRAIN_CLEANED_VAL, PRETRAIN_CLEANED_TEST, PSCDB_CLEANED_VAL, PSCDB_CLEANED_TEST, \
    VAL_SIZE_PSCDB, TEST_SIZE_PSCDB, VAL_SIZE_PRETRAIN, TEST_SIZE_PRETRAIN, RANDOM_SEED, PSCDB_PDBS_SUFFIX, \
    PATH_PDBS_DIR, PSCDB_CLASS_WEIGHTS, OTHER_MOTION_PROTEINS_ORIGINAL_FORMAT_PATH
from preprocessing.dataset import create_dataset_pscdb, create_dataset_pretrain, load_dataset
from preprocessing.utils import pscdb_read, get_uniprot_IDs_and_pdb_codes, train_test_validation_split, \
    get_pdb_paths_pscdb, read_others_original_format


__INTEGRATE_OTHER_TYPE_PROTEINS: final = True
__RECREATE_PRETRAINING: final = False


def main():
    # Read raw data
    df = pscdb_read(path=PSCDB_PATH, drop_duplicate_pdb_codes=True)
    df2 = df.iloc[0:-1]

    uniprots, pdbs, pdb_paths = get_uniprot_IDs_and_pdb_codes(PATH_PDBS_JSON)

    # Perform train/validation/test split on PSCDB
    df_train, df_val, df_test = train_test_validation_split(
        dataset=df2,
        val_size=VAL_SIZE_PSCDB,
        test_size=TEST_SIZE_PSCDB,
        random_seed=RANDOM_SEED
    )
    df_train_other_motion, df_val_other_motion, df_test_other_motion = read_others_original_format(
        OTHER_MOTION_PROTEINS_ORIGINAL_FORMAT_PATH,
        val_size=VAL_SIZE_PSCDB,
        test_size=TEST_SIZE_PSCDB,
        random_seed=RANDOM_SEED
    )

    # Integrate the "other_motion" proteins from the original format dataset
    if __INTEGRATE_OTHER_TYPE_PROTEINS:
        df_train = pd.concat([df_train, df_train_other_motion])
        df_val = pd.concat([df_val, df_val_other_motion])
        df_test = pd.concat([df_test, df_test_other_motion])

    # Perform train/validation/test split on pre-training proteins (PSCDB-ones excluded)
    pdb_paths_train, pdb_paths_val, pdb_paths_test = train_test_validation_split(
        pdb_paths,
        val_size=VAL_SIZE_PRETRAIN,
        test_size=TEST_SIZE_PRETRAIN,
        random_seed=RANDOM_SEED
    )

    # Get the PDB paths of the PSCDB train/validation/test proteins
    '''
    pscdb_pdb_paths_train = get_pdb_paths_pscdb(df_train, os.path.join(PSCDB_CLEANED_TRAIN, PSCDB_PDBS_SUFFIX))
    pscdb_pdb_paths_val = get_pdb_paths_pscdb(df_val, os.path.join(PSCDB_CLEANED_VAL, PSCDB_PDBS_SUFFIX))
    pscdb_pdb_paths_test = get_pdb_paths_pscdb(df_test, os.path.join(PSCDB_CLEANED_TEST, PSCDB_PDBS_SUFFIX))
    '''
    pscdb_pdb_paths_train = get_pdb_paths_pscdb(df_train, PATH_PDBS_DIR)
    pscdb_pdb_paths_val = get_pdb_paths_pscdb(df_val, PATH_PDBS_DIR)
    pscdb_pdb_paths_test = get_pdb_paths_pscdb(df_test, PATH_PDBS_DIR)

    # Add PSCDB train/validation/test PDB paths to the respective pre-training PDB path lists
    pdb_paths_train = pdb_paths_train + pscdb_pdb_paths_train
    pdb_paths_val = pdb_paths_val + pscdb_pdb_paths_val
    pdb_paths_test = pdb_paths_test + pscdb_pdb_paths_test

    # Create PSCDB classification datasets
    ds_cl_train = create_dataset_pscdb(df_train, export_path=PSCDB_CLEANED_TRAIN, in_memory=False, store_params=True)
    ds_cl_val = create_dataset_pscdb(df_val, export_path=PSCDB_CLEANED_VAL, in_memory=False, store_params=True)
    ds_cl_test = create_dataset_pscdb(df_test, export_path=PSCDB_CLEANED_TEST, in_memory=False, store_params=True)

    # Copy PSCDB PDB files to AlphaFold directory, otherwise pre-train dataset creation won't work cuz: "graphein cool!"
    copy_all_pscdb_files = input("Copy all PSCDB .pdb files to alphafold directory (0: no, 1: yes)? ")
    if int(copy_all_pscdb_files) != 0:
        shutil.copytree(src=os.path.join(PSCDB_CLEANED_TRAIN, PSCDB_PDBS_SUFFIX), dst=PATH_PDBS_DIR, dirs_exist_ok=True)
        shutil.copytree(src=os.path.join(PSCDB_CLEANED_VAL, PSCDB_PDBS_SUFFIX), dst=PATH_PDBS_DIR, dirs_exist_ok=True)
        shutil.copytree(src=os.path.join(PSCDB_CLEANED_TEST, PSCDB_PDBS_SUFFIX), dst=PATH_PDBS_DIR, dirs_exist_ok=True)

    # Create pre-training datasets
    if __RECREATE_PRETRAINING:
        ds_pt_train = create_dataset_pretrain(
            pdb_paths=pdb_paths_train,
            export_path=PRETRAIN_CLEANED_TRAIN,
            in_memory=False,
            store_params=True
        )
        ds_pt_val = create_dataset_pretrain(
            pdb_paths=pdb_paths_val,
            export_path=PRETRAIN_CLEANED_VAL,
            in_memory=False,
            store_params=True
        )
        ds_pt_test = create_dataset_pretrain(
            pdb_paths=pdb_paths_test,
            export_path=PRETRAIN_CLEANED_TEST,
            in_memory=False,
            store_params=True
        )

        # Create data loader to check if everything's ok
        dl = DataLoader(ds_pt_train, batch_size=2, shuffle=True, drop_last=True)
        print(len(dl))
        print(next(iter(dl)))

        # Create data loader to check if everything's ok
        dl = DataLoader(ds_pt_val, batch_size=2, shuffle=True, drop_last=True)
        print(len(dl))
        print(next(iter(dl)))

        # Create data loader to check if everything's ok
        dl = DataLoader(ds_pt_test, batch_size=2, shuffle=True, drop_last=True)
        print(len(dl))
        print(next(iter(dl)))

    # Create data loader to check if everything's ok
    dl = DataLoader(ds_cl_train, batch_size=1, shuffle=True, drop_last=True)
    min_n = 1000000000
    max_n = 0
    n_nodes = []
    y_distribution_train = {}
    for el in iter(dl):
        if el.num_nodes < min_n:
            min_n = el.num_nodes
        if el.num_nodes > max_n:
            max_n = el.num_nodes
        if int(el.y) not in y_distribution_train:
            y_distribution_train[int(el.y)] = 1
        else:
            y_distribution_train[int(el.y)] += 1
        n_nodes.append(el.num_nodes)
    print(len(dl))
    print(next(iter(dl)))

    # Create data loader to check if everything's ok
    dl = DataLoader(ds_cl_val, batch_size=1, shuffle=False, drop_last=True)
    y_distribution_val = {}
    for el in iter(dl):
        if el.num_nodes < min_n:
            min_n = el.num_nodes
        if el.num_nodes > max_n:
            max_n = el.num_nodes
        if int(el.y) not in y_distribution_val:
            y_distribution_val[int(el.y)] = 1
        else:
            y_distribution_val[int(el.y)] += 1
        n_nodes.append(el.num_nodes)
    print(len(dl))
    print(next(iter(dl)))

    # Create data loader to check if everything's ok
    dl = DataLoader(ds_cl_test, batch_size=1, shuffle=False, drop_last=True)
    y_distribution_test = {}
    for el in iter(dl):
        if el.num_nodes < min_n:
            min_n = el.num_nodes
            print(min_n)
        if el.num_nodes > max_n:
            max_n = el.num_nodes
        if int(el.y) not in y_distribution_test:
            y_distribution_test[int(el.y)] = 1
        else:
            y_distribution_test[int(el.y)] += 1
        n_nodes.append(el.num_nodes)

    print(f"Min is: {min_n}")
    print(f"Max is: {max_n}")
    print(f"Median is: {np.median(n_nodes)}")
    print(f"Mean is: {np.mean(n_nodes)}")
    print(f"Quantiles: {np.quantile(n_nodes, q=[0, 0.25, 0.5, 0.75, 1])}")
    print(len(dl))
    print(next(iter(dl)))
    print(f"Class distribution train: {y_distribution_train}")
    print(f"Class distribution train: {y_distribution_val}")
    print(f"Class distribution train: {y_distribution_test}")
    y_distribution: dict[int, int] = y_distribution_train.copy()
    for cl in y_distribution:
        y_distribution[cl] += y_distribution_val[cl] + y_distribution_test[cl]
    print(f"Total class  distribution: {y_distribution}")
    max_cl = max(y_distribution.values())
    class_weights = [0 for _ in range(0, len(y_distribution))]
    for cl in y_distribution:
        class_weights[cl] = float(max_cl / y_distribution[cl])
    print(f"Class weights: {class_weights}")
    class_weights = torch.tensor(class_weights)
    torch.save(class_weights, PSCDB_CLASS_WEIGHTS)
    class_weights = torch.load(PSCDB_CLASS_WEIGHTS)
    print(f"Loaded class weights {class_weights}")

    # Load the dataset and create the data loader to check if everything's ok
    #ds2 = load_dataset(PRETRAIN_CLEANED_TRAIN, dataset_type="pretrain")
    #print(len(ds2))
    #dl = DataLoader(ds2, batch_size=2, shuffle=True, drop_last=True)
    #print(next(iter(dl)))

    ds3 = load_dataset(PSCDB_CLEANED_TRAIN, dataset_type="pscdb")
    dl = DataLoader(ds3, batch_size=2, shuffle=False, drop_last=True)
    print(next(iter(dl)))


if __name__ == '__main__':
    main()
