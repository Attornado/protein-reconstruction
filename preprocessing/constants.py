from typing import final
import os
import json


# Generic dataset-related constants
DATA_PATH: final = "data"


# Additional PDBs from AlphaFold-related constants
UNIPROTS_KEY: final = "uniprotIDs"
PDBS_KEY: final = "pdbIDs"
PATHS_KEY: final = "pdbPaths"
PATH_ALPHAFOLD: final = os.path.join(DATA_PATH, "alphafold")
PATH_PDBS_JSON: final = os.path.join(PATH_ALPHAFOLD, "pdbs.json")
PATH_PDBS_DIR: final = os.path.join(PATH_ALPHAFOLD, "pdbs")


# PSCDB-related constants
MOTION_TYPE: final = "motion"
PDB: final = "pdb"
USED_COLUMNS: final = {"Free PDB": PDB, "motion_type": MOTION_TYPE}
OTHER_MOTION_TYPE: final = "other_motion"
PSCDB_PATH: final = os.path.join(DATA_PATH, "pscdb", "structural_rearrangement_data.csv")


# PSCDB original format dataset-related constants
PSCDB_ORIGINAL_FORMAT_PATH: final = os.path.join(DATA_PATH, "pscdb", "pscdb.csv")
OTHER_MOTION_PROTEINS_ORIGINAL_FORMAT_PATH: final = os.path.join(DATA_PATH, "pscdb", "other_motion_proteins.csv")
OBSOLETE_STRUCTURES: final = frozenset(['1m80', '1cmw', '1g40', '2ihi', '1hl0', '2gkq', '2glb', '2g2j', '2dpo', '2h98',
                                        '2gu9', '2bg1', '1q4o', '1il5', '3cey', '1yks'])
OTHER_MOTION_COLUMN_NAMES: final = frozenset(["Other/ Plastic", "Other/ domain-like", "Other/ local-like"])
MOTION_COLUMN: final = "Type of motion"
FREE_PDB_COLUMN: final = "FreeID"
BOUND_PDB_COLUMN: final = "BoundID"


# Cleaned dataset-related constants
CLEANED_DATA: final = os.path.join(DATA_PATH, "cleaned")

PRETRAIN_CLEANED: final = os.path.join(CLEANED_DATA, "pretraining")
PRETRAIN_CLEANED_TRAIN: final = os.path.join(PRETRAIN_CLEANED, "train")
PRETRAIN_CLEANED_VAL: final = os.path.join(PRETRAIN_CLEANED, "validation")
PRETRAIN_CLEANED_TEST: final = os.path.join(PRETRAIN_CLEANED, "test")

PSCDB_CLEANED: final = os.path.join(CLEANED_DATA, "pscdb")
PSCDB_CLEANED_TRAIN: final = os.path.join(PSCDB_CLEANED, "train")
PSCDB_CLEANED_VAL: final = os.path.join(PSCDB_CLEANED, "validation")
PSCDB_CLEANED_TEST: final = os.path.join(PSCDB_CLEANED, "test")
PSCDB_CLASS_WEIGHTS: final = os.path.join(PSCDB_CLEANED, "class_weights.pt")

PSCDB_PDBS_SUFFIX: final = "raw"
PARAMS_DIR_SUFFIX: final = "params"
PARAMS_CSV_SUFFIX: final = "param_df.csv"
PARAMS_JSON_SUFFIX: final = "params.json"


# Randomness-related constants
RANDOM_SEED: final = 42


# Split-related constants
VAL_SIZE_PSCDB: final = 0.15
TEST_SIZE_PSCDB: final = 0.15
VAL_SIZE_PRETRAIN: final = 0.20
TEST_SIZE_PRETRAIN: final = 0.20


# Machine-specific config-related constants
CONFIG_FILES_PATH: final = "config"
HARDWARE_CONFIG_PATH: final = os.path.join(CONFIG_FILES_PATH, "hardware.json")

# Read hardware config file
with open(HARDWARE_CONFIG_PATH, "r") as fp:
    config = json.load(fp)
    NUM_CORES: final = config["num_cores"]
    RAM_SIZE: final = config["ram_size"]
    NUM_GPUS: final = config["num_gpus"]
    VRAM_SIZE: final = config["vram_size"]
    del config
