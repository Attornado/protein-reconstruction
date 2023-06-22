import argparse
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
    PATH_PDBS_DIR, PSCDB_CLASS_WEIGHTS, OTHER_MOTION_PROTEINS_ORIGINAL_FORMAT_PATH, PSCDB_PAIRED_CLEANED_VAL, \
    PSCDB_PAIRED_CLEANED_TEST, PSCDB_PAIRED_CLEANED_TRAIN, ENZYMES_CLASSIFICATION_CSV, FOLD_CLASSIFICATION_CSV, \
    VAL_SIZE_ENZYMES, TEST_SIZE_ENZYMES, ENZYMES_CLASS_WEIGHTS, FOLD_CLASSIFICATION_CLASS_WEIGHTS, \
    ENZYMES_CLEANED_TRAIN, ENZYMES_CLEANED_VAL, ENZYMES_CLEANED_TEST, FOLD_CLASSIFICATION_CLEANED_TRAIN, \
    FOLD_CLASSIFICATION_CLEANED_VAL, FOLD_CLASSIFICATION_CLEANED_TEST, VAL_SIZE_FOLD, TEST_SIZE_FOLD
from preprocessing.dataset.dataset_creation import create_dataset_pscdb, create_dataset_pretrain, load_dataset, \
    create_dataset_pscdb_paired, create_dataset_enzymes, create_dataset_fold_classification
from preprocessing.dataset.paired_dataset import PairedDataLoader
from preprocessing.utils import pscdb_read, get_uniprot_IDs_and_pdb_codes, train_test_validation_split, \
    get_pdb_paths_pscdb, read_others_original_format


__INTEGRATE_OTHER_TYPE_PROTEINS: final = True
__RECREATE_PSCDB: final = True
__RECREATE_PRETRAINING: final = False
__RECREATE_ENZYMES: final = True
__RECREATE_FOLD: final = True


def main(args):

    if args.recreate_pscdb:
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
        if args.integrate_missing_proteins:
            df_train = pd.concat([df_train, df_train_other_motion])
            df_val = pd.concat([df_val, df_val_other_motion])
            df_test = pd.concat([df_test, df_test_other_motion])

        # Perform train/validation/test split on pre-training proteins (PSCDB-ones excluded)
        pdb_paths_train, pdb_paths_val, pdb_paths_test = train_test_validation_split(
            pdb_paths,
            val_size=VAL_SIZE_PRETRAIN,
            test_size=TEST_SIZE_PRETRAIN,
            random_seed=args.seed
        )

        # Get the PDB paths of the PSCDB train/validation/test proteins
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

        ds_cl_train2 = create_dataset_pscdb_paired(df_train, export_path=PSCDB_PAIRED_CLEANED_TRAIN, store_params=True)
        ds_cl_val2 = create_dataset_pscdb_paired(df_val, export_path=PSCDB_PAIRED_CLEANED_VAL, store_params=True)
        ds_cl_test2 = create_dataset_pscdb_paired(df_test, export_path=PSCDB_PAIRED_CLEANED_TEST, store_params=True)

        # Copy PSCDB PDB files to AlphaFold directory, otherwise pre-train dataset creation won't work cuz:
        # "Graphein cool!"
        # copy_all_pscdb_files = input("Copy all PSCDB .pdb files to alphafold directory (0: no, 1: yes)? ")
        copy_all_pscdb_files = args.copy_all_pscdb_files
        if int(copy_all_pscdb_files) != 0:
            shutil.copytree(src=os.path.join(PSCDB_CLEANED_TRAIN, PSCDB_PDBS_SUFFIX), dst=PATH_PDBS_DIR,
                            dirs_exist_ok=True)
            shutil.copytree(src=os.path.join(PSCDB_CLEANED_VAL, PSCDB_PDBS_SUFFIX), dst=PATH_PDBS_DIR,
                            dirs_exist_ok=True)
            shutil.copytree(src=os.path.join(PSCDB_CLEANED_TEST, PSCDB_PDBS_SUFFIX), dst=PATH_PDBS_DIR,
                            dirs_exist_ok=True)

        # Create pre-training datasets
        if args.recreate_pretraining:
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
        print(len(dl))
        print(next(iter(dl)))

        print(f"Min is: {min_n}")
        print(f"Max is: {max_n}")
        print(f"Median is: {np.median(n_nodes)}")
        print(f"Mean is: {np.mean(n_nodes)}")
        print(f"Quantiles: {np.quantile(n_nodes, q=[0, 0.25, 0.5, 0.75, 1])}")
        print(len(dl))
        print(next(iter(dl)))
        print(f"Class distribution train: {y_distribution_train}")
        print(f"Class distribution val: {y_distribution_val}")
        print(f"Class distribution test: {y_distribution_test}")
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

        # Create data loader to check if everything's ok
        dl = PairedDataLoader(ds_cl_train2, batch_size=1, shuffle=True, drop_last=True)
        min_n = 1000000000
        max_n = 0
        n_nodes = []
        y_distribution_train = {}
        for el in iter(dl):
            if el.a[0].num_nodes < min_n:
                min_n = el.a[0].num_nodes
            if el.b[0].num_nodes < min_n:
                min_n = el.b[0].num_nodes
            if el.a[0].num_nodes > max_n:
                max_n = el.a[0].num_nodes
            if el.b[0].num_nodes > max_n:
                max_n = el.b[0].num_nodes
            if int(el.a[0].y) not in y_distribution_train:
                y_distribution_train[int(el.a[0].y)] = 1
            else:
                y_distribution_train[int(el.a[0].y)] += 1
            n_nodes.append(el.a[0].num_nodes)
            n_nodes.append(el.b[0].num_nodes)
        print(len(dl))
        el = next(iter(dl))
        print(f"Before: {el.a[0]}")
        print(f"After: {el.b[0]}")

        # Create data loader to check if everything's ok
        dl = PairedDataLoader(ds_cl_val2, batch_size=1, shuffle=False, drop_last=True)
        y_distribution_val = {}
        for el in iter(dl):
            if el.a[0].num_nodes < min_n:
                min_n = el.a[0].num_nodes
            if el.b[0].num_nodes < min_n:
                min_n = el.b[0].num_nodes
            if el.a[0].num_nodes > max_n:
                max_n = el.a[0].num_nodes
            if el.b[0].num_nodes > max_n:
                max_n = el.b[0].num_nodes
            if int(el.a[0].y) not in y_distribution_val:
                y_distribution_val[int(el.a[0].y)] = 1
            else:
                y_distribution_val[int(el.a[0].y)] += 1
            n_nodes.append(el.a[0].num_nodes)
            n_nodes.append(el.b[0].num_nodes)
        print(len(dl))
        el = next(iter(dl))
        print(f"Before: {el.a[0]}")
        print(f"After: {el.b[0]}")

        # Create data loader to check if everything's ok
        dl = PairedDataLoader(ds_cl_test2, batch_size=1, shuffle=False, drop_last=True)
        y_distribution_test = {}
        for el in iter(dl):
            if el.a[0].num_nodes < min_n:
                min_n = el.a[0].num_nodes
            if el.b[0].num_nodes < min_n:
                min_n = el.b[0].num_nodes
            if el.a[0].num_nodes > max_n:
                max_n = el.a[0].num_nodes
            if el.b[0].num_nodes > max_n:
                max_n = el.b[0].num_nodes
            if int(el.a[0].y) not in y_distribution_test:
                y_distribution_test[int(el.a[0].y)] = 1
            else:
                y_distribution_test[int(el.a[0].y)] += 1
            n_nodes.append(el.a[0].num_nodes)
            n_nodes.append(el.b[0].num_nodes)
        print(len(dl))
        el = next(iter(dl))
        print(f"Before: {el.a[0]}")
        print(f"After: {el.b[0]}")

        print(f"Min is: {min_n}")
        print(f"Max is: {max_n}")
        print(f"Median is: {np.median(n_nodes)}")
        print(f"Mean is: {np.mean(n_nodes)}")
        print(f"Quantiles: {np.quantile(n_nodes, q=[0, 0.25, 0.5, 0.75, 1])}")
        print(len(dl))
        print(next(iter(dl)))
        print(f"Class distribution train: {y_distribution_train}")
        print(f"Class distribution val: {y_distribution_val}")
        print(f"Class distribution test: {y_distribution_test}")
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
        if args.recreate_pretraining:
            ds2 = load_dataset(PRETRAIN_CLEANED_TRAIN, dataset_type="pretrain")
            print(len(ds2))
            dl = DataLoader(ds2, batch_size=2, shuffle=True, drop_last=True)
            print(next(iter(dl)))

        ds3 = load_dataset(PSCDB_CLEANED_TRAIN, dataset_type="pscdb")
        dl = DataLoader(ds3, batch_size=2, shuffle=False, drop_last=True)
        print(next(iter(dl)))

        ds3 = load_dataset(PSCDB_PAIRED_CLEANED_TRAIN, dataset_type="pscdb_paired")
        dl = PairedDataLoader(ds3, batch_size=2, shuffle=False, drop_last=True)
        print(next(iter(dl)))

    if args.recreate_enzymes:
        # Read raw data
        df = pd.read_csv(ENZYMES_CLASSIFICATION_CSV, index_col=False)
        df2 = df.iloc[0:-1]

        # Perform train/validation/test split on PSCDB
        df_train, df_val, df_test = train_test_validation_split(
            dataset=df2,
            val_size=VAL_SIZE_ENZYMES,
            test_size=TEST_SIZE_ENZYMES,
            random_seed=RANDOM_SEED
        )

        # Create ENZYMES classification datasets
        ds_cl_train = create_dataset_enzymes(df_train, export_path=ENZYMES_CLEANED_TRAIN, in_memory=False,
                                             store_params=True)
        ds_cl_val = create_dataset_enzymes(df_val, export_path=ENZYMES_CLEANED_VAL, in_memory=False, store_params=True)
        ds_cl_test = create_dataset_enzymes(df_test, export_path=ENZYMES_CLEANED_TEST, in_memory=False,
                                            store_params=True)

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
        print(len(dl))
        print(next(iter(dl)))

        print(f"Min is: {min_n}")
        print(f"Max is: {max_n}")
        print(f"Median is: {np.median(n_nodes)}")
        print(f"Mean is: {np.mean(n_nodes)}")
        print(f"Quantiles: {np.quantile(n_nodes, q=[0, 0.25, 0.5, 0.75, 1])}")
        print(len(dl))
        print(next(iter(dl)))
        print(f"Class distribution train: {y_distribution_train}")
        print(f"Class distribution val: {y_distribution_val}")
        print(f"Class distribution test: {y_distribution_test}")
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
        torch.save(class_weights, ENZYMES_CLASS_WEIGHTS)
        class_weights = torch.load(ENZYMES_CLASS_WEIGHTS)
        print(f"Loaded class weights {class_weights}")

    if args.recreate_fold:
        # Read raw data
        df = pd.read_csv(FOLD_CLASSIFICATION_CSV, index_col=False)
        df2 = df.iloc[0:-1]

        # Perform train/validation/test split on PSCDB
        df_train, df_val, df_test = train_test_validation_split(
            dataset=df2,
            val_size=VAL_SIZE_FOLD,
            test_size=TEST_SIZE_FOLD,
            random_seed=RANDOM_SEED
        )

        # Create ENZYMES classification datasets
        ds_cl_train = create_dataset_fold_classification(df_train, export_path=FOLD_CLASSIFICATION_CLEANED_TRAIN,
                                                         in_memory=False, store_params=True)
        ds_cl_val = create_dataset_fold_classification(df_val, export_path=FOLD_CLASSIFICATION_CLEANED_VAL,
                                                       in_memory=False, store_params=True)
        ds_cl_test = create_dataset_fold_classification(df_test, export_path=FOLD_CLASSIFICATION_CLEANED_TEST,
                                                        in_memory=False, store_params=True)

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
        print(len(dl))
        print(next(iter(dl)))

        print(f"Min is: {min_n}")
        print(f"Max is: {max_n}")
        print(f"Median is: {np.median(n_nodes)}")
        print(f"Mean is: {np.mean(n_nodes)}")
        print(f"Quantiles: {np.quantile(n_nodes, q=[0, 0.25, 0.5, 0.75, 1])}")
        print(len(dl))
        print(next(iter(dl)))
        print(f"Class distribution train: {y_distribution_train}")
        print(f"Class distribution val: {y_distribution_val}")
        print(f"Class distribution test: {y_distribution_test}")
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
        torch.save(class_weights, FOLD_CLASSIFICATION_CLASS_WEIGHTS)
        class_weights = torch.load(FOLD_CLASSIFICATION_CLASS_WEIGHTS)
        print(f"Loaded class weights {class_weights}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='the integer value seed for global random state in Lightning')
    parser.add_argument('--integrate_missing_proteins', type=bool, default=__INTEGRATE_OTHER_TYPE_PROTEINS,
                        help='add missing proteins to PSCDB')
    parser.add_argument('--recreate_pscdb', type=bool, default=__RECREATE_PSCDB, help='recreate the PSCDB dataset')
    parser.add_argument('--recreate_pretraining', type=bool, default=__RECREATE_PRETRAINING,
                        help='recreate the pretrain dataset')
    parser.add_argument('--recreate_enzymes', type=bool, default=__RECREATE_ENZYMES, help='recreate ENZYMES dataset')
    parser.add_argument('--recreate_fold', type=bool, default=__RECREATE_FOLD,
                        help='recreate the fold classification dataset')
    parser.add_argument('--copy_all_pscdb_files', type=int, default=0, choices=[0, 1],
                        help='copy all PSCDB PDB files into pretraining data directory')
    arguments = parser.parse_args()
    main(args=arguments)
