import collections
import json
import os
from typing import Union, List, Optional
import pandas as pd
from preprocessing.constants import UNIPROTS_KEY, PDBS_KEY, USED_COLUMNS, RANDOM_SEED, TEST_SIZE_PSCDB, \
    VAL_SIZE_PSCDB, PDB, PATHS_KEY, MOTION_COLUMN, FREE_PDB_COLUMN, MOTION_TYPE, OTHER_MOTION_TYPE, \
    OTHER_MOTION_COLUMN_NAMES, BOUND_PDB_COLUMN, PDB_BOUND
from sklearn.model_selection import train_test_split


def get_uniprot_IDs_and_pdb_codes(path: str) -> tuple[list[str], list[str], list[str]]:
    """
    This function takes in a path to a JSON file containing the UniProt IDs and PDB codes, reading them from the JSON
    file and returning them.

    :param path: the path to the json file containing the UniProt IDs and PDB codes.
    :type path: str
    :return: a tuple of two lists containing the UniProt IDs and PDB codes.
    """

    with open(path, "r") as fp:
        data = json.load(fp)
        uniprotIDs = data[UNIPROTS_KEY]
        pdbIDs = data[PDBS_KEY]
        paths = data[PATHS_KEY]
        return uniprotIDs, pdbIDs, paths


def pscdb_read(path: str, drop_duplicate_pdb_codes: bool = True) -> pd.DataFrame:
    """
    Reads the CSV file at the given path, drops all columns except the ones we want, and renames the columns to the
    names we want.

    :param path: the path to the csv file
    :type path: str
    :param drop_duplicate_pdb_codes: if True, then duplicate PDB codes will be dropped when the dataframe is read,
        defaults to True.
    :type drop_duplicate_pdb_codes: bool
    :return: A dataframe corresponding to the PSCDB dataset.
    """
    df = pd.read_csv(path, index_col=False)
    df = df.drop(df.columns.difference(USED_COLUMNS.keys()), axis=1)
    df = df.rename(columns=USED_COLUMNS)

    # Remove duplicates if required
    if drop_duplicate_pdb_codes:
        df = get_unique_pdbs(df)

    return df


def get_unique_pdbs(pscdb: pd.DataFrame) -> pd.DataFrame:
    """
    Gets unique rows from PSCDB dataframe, with respect to the PDB-code column.

    :param pscdb: the dataframe containing the PSCDB data
    :type pscdb: pd.DataFrame

    :return: the pscdb dataframe without duplicate PDB codes.
    """
    return pscdb.drop_duplicates(subset=[PDB], keep="first")


def get_pdb_paths_pscdb(pscdb: pd.DataFrame, root_path: str) -> List[str]:
    """
    Given a PSCDB dataframe and a root path, return a list of paths to the PDB files.

    :param pscdb: the dataframe containing the PSCDB data
    :type pscdb: pd.DataFrame
    :param root_path: the path to the directory containing the PDB files
    :type root_path: str
    :return: A list of paths to the PDB files.
    """
    pdb_paths = pscdb[PDB].to_list()
    for i in range(0, len(pdb_paths)):
        pdb_paths[i] = os.path.join(root_path, pdb_paths[i] + ".pdb")
    return pdb_paths


def train_test_validation_split(dataset: Union[pd.DataFrame, List[str]], val_size: float = VAL_SIZE_PSCDB,
                                test_size: float = TEST_SIZE_PSCDB, random_seed: int = RANDOM_SEED) -> \
        tuple[Union[pd.DataFrame, List[str]], Union[pd.DataFrame, List[str]], Union[pd.DataFrame, List[str]]]:
    """
    Splits a dataframe into train, validation and test sets.

    :param dataset: the dataframe to split
    :type dataset: pd.DataFrame
    :param val_size: the ratio of the validation set to the entire dataset
    :type val_size: float
    :param test_size: the ratio of the test set to the entire dataset
    :type test_size: float
    :param random_seed: The random seed to use for the split
    :type random_seed: int

    :return: A tuple of three dataframes, representing the train, validation and test sets, respectively.
    :rtype: tuple[Union[pd.DataFrame, List[str]], Union[pd.DataFrame, List[str]], Union[pd.DataFrame, List[str]]]
    """

    if type(dataset) == list:
        df = pd.DataFrame(dataset)
    else:
        df = dataset

    df_train, df_val = train_test_split(df, test_size=val_size, random_state=random_seed)
    df_train, df_test = train_test_split(df_train, test_size=val_size / (1 - test_size), random_state=random_seed)

    if type(dataset) == list:
        return df_train[0].to_list(), df_val[0].to_list(), df_test[0].to_list()
    else:
        return df_train, df_val, df_test


class FrozenDict(collections.Mapping):
    """
    Simple class representing a dictionary that cannot be changed.
    """

    def __init__(self, *args, **kwargs):
        """
        Constructs a new frozen dictionary from the given arguments.
        """
        self._d = dict(*args, **kwargs)
        self._hash = None

    def __str__(self):
        return str(self._d)

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def keys(self):
        return self._d.keys()

    def values(self):
        return self._d.values()

    def items(self):
        return self._d.items()

    def __hash__(self):
        # It would have been simpler and maybe more obvious to
        # use hash(tuple(sorted(self._d.iteritems()))) from this discussion
        # so far, but this solution is O(n). I don't know what kind of
        # n we are going to run into, but sometimes it's hard to resist the
        # urge to optimize when it will gain improved algorithmic performance.
        if self._hash is None:
            hash_ = 0
            for pair in self.items():
                hash_ ^= hash(pair)
            self._hash = hash_
        return self._hash


def read_others_original_format(path: str, val_size: Optional[float] = None, test_size: Optional[float] = None,
                                random_seed: int = RANDOM_SEED, split_from_others_only: bool = True) -> \
        Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:

    # Read csv containing "other_motion" structures
    df = pd.read_csv(path, index_col=False)

    # If complete pscdb data in original format is given, then remove it all except from the "other_motion" proteins
    if not split_from_others_only:
        sub_dfs: list[pd.DataFrame] = []
        for name in OTHER_MOTION_COLUMN_NAMES:
            sub_dfs.append(df[df[MOTION_COLUMN] == name].drop_duplicates(subset=[FREE_PDB_COLUMN]))
        df = pd.concat(sub_dfs)

    # Drop all columns besides PDB and motion type ones
    df = df.drop(df.columns.difference([FREE_PDB_COLUMN, BOUND_PDB_COLUMN, MOTION_COLUMN]), axis=1)
    df = df.drop_duplicates(subset=[FREE_PDB_COLUMN])

    # Rename columns according to our format
    column_renaming = {
        FREE_PDB_COLUMN: PDB,
        BOUND_PDB_COLUMN: PDB_BOUND,
        MOTION_COLUMN: MOTION_TYPE
    }
    df = df.rename(columns=column_renaming)

    # Replace <PDBcode>_<chains> with <PDBcode>
    df.loc[:, PDB] = df[PDB].apply(lambda free_pdb_id: free_pdb_id.split("_")[0].strip())
    df.loc[:, PDB_BOUND] = df[PDB_BOUND].apply(lambda free_pdb_id: free_pdb_id.split("_")[0].strip())
    df.loc[:, MOTION_TYPE] = OTHER_MOTION_TYPE

    # Perform train/validation/test split if sizes are given
    if val_size is None and test_size is None:
        return df
    else:
        if val_size is None:
            val_size = test_size
        elif test_size is None:
            test_size = val_size
        train_df, val_df, test_df = train_test_validation_split(df, val_size=val_size, test_size=test_size,
                                                                random_seed=random_seed)
        return train_df, val_df, test_df
