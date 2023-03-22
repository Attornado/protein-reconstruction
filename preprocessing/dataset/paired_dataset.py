from typing import Optional, List, Callable, Union, final
import torch
from graphein.ml import GraphFormatConvertor, ProteinGraphDataset
from graphein.protein import ProteinGraphConfig
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater


class PairedData(Data):

    __A_KEY: final = 'a'
    __B_KEY: final = 'b'

    def __init__(self, a: Data, b: Data, global_y: Optional[torch.Tensor] = None, **kwargs):
        """
        Pairs two graphs together in a single ``Data`` instance.
        The first graph is accessed via ``data.a`` (e.g. ``data.a.x``) and the second via ``data.b``.

        :param a: The first graph.
        :type a: torch_geometric.data.Data
        :param b: The second graph.
        :type b: torch_geometric.data.Data
        :param global_y: global y value for the pair data (optional, default None)
        :param **kwargs: Additional (optional) attributes

        :return: The paired graph.
        """
        super().__init__(x=None, edge_index=None, edge_attr=None, y=global_y, pos=None, **kwargs)
        self.a = a
        self.b = b

    @property
    def a(self) -> Data:
        """
        Gets the first graph of the pair.
        """
        return self[self.__A_KEY]

    @a.setter
    def a(self, a: Data):
        """
        Sets the first graph of the pair.
        """
        self[self.__A_KEY] = a

    @property
    def b(self) -> Data:
        """
        Gets the second graph of the pair.
        """
        return self[self.__B_KEY]

    @b.setter
    def b(self, a: Data):
        """
        Sets the second graph of the pair.
        """
        self[self.__B_KEY] = a


class PairedProteinGraphDataset(Dataset):

    def __init__(self,
                 root: str,
                 pdb_paths0: Optional[List[str]] = None,
                 pdb_paths1: Optional[List[str]] = None,
                 pdb_codes0: Optional[List[str]] = None,
                 pdb_codes1: Optional[List[str]] = None,
                 uniprot_ids0: Optional[List[str]] = None,
                 uniprot_ids1: Optional[List[str]] = None,
                 graph_labels0: Optional[List[torch.Tensor]] = None,
                 graph_labels1: Optional[List[torch.Tensor]] = None,
                 node_labels0: Optional[List[torch.Tensor]] = None,
                 node_labels1: Optional[List[torch.Tensor]] = None,
                 chain_selections0: Optional[List[str]] = None,
                 chain_selections1: Optional[List[str]] = None,
                 graphein_config: ProteinGraphConfig = ProteinGraphConfig(),
                 graph_format_convertor: GraphFormatConvertor = GraphFormatConvertor(
                     src_format="nx", dst_format="pyg"
                 ),
                 graph_transformation_funcs: Optional[List[Callable]] = None,
                 pdb_transform: Optional[List[Callable]] = None,
                 paired_transform: Optional[Callable] = None,
                 paired_pre_transform: Optional[Callable] = None,
                 paired_pre_filter: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 num_cores: int = 16,
                 af_version: int = 2):
        """Dataset class for protein graphs pairs.

        Dataset base class for creating graph datasets.
        See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
        create_dataset.html>`__ for the accompanying tutorial.

        :param root: Root directory where the dataset should be saved.
        :type root: str
        :param pdb_paths0: First list of full path of pdb files to load. Defaults to
            ``None``.
        :type pdb_paths0: Optional[List[str]], optional
        :param pdb_paths1: Second list of full path of pdb files to load. Defaults to
            ``None``.
        :type pdb_paths1: Optional[List[str]], optional
        :param pdb_codes0: First list of PDB codes to download and parse from the PDB.
            Defaults to ``None``.
        :type pdb_codes0: Optional[List[str]], optional
        :param pdb_codes1: Second list of PDB codes to download and parse from the PDB.
            Defaults to ``None``.
        :type pdb_codes1: Optional[List[str]], optional
        :param uniprot_ids0: First list of Uniprot IDs to download and parse from
            Alphafold Database. Defaults to ``None``.
        :type uniprot_ids0: Optional[List[str]], optional
        :param uniprot_ids1: Second list of Uniprot IDs to download and parse from
            Alphafold Database. Defaults to ``None``.
        :type uniprot_ids1: Optional[List[str]], optional
        :param graph_labels0: First list of graph-level labels. Defaults to ``None``.
        :type graph_labels0: Optional[List[torch.Tensor]], optional
        :param graph_labels1: Second list of graph-level labels. Defaults to ``None``.
        :type graph_labels1: Optional[List[torch.Tensor]], optional
        :param node_labels0: First list of node-level labels. Defaults to ``None``.
        :type node_labels0: Optional[List[torch.Tensor]], optional
        :param node_labels1: Second list of node-level labels. Defaults to ``None``.
        :type node_labels1: Optional[List[torch.Tensor]], optional
        :param chain_selections0: First chain selection list, defaults to ``None``.
        :type chain_selections0: Optional[List[str]], optional
        :param chain_selections1: Second chain selection list, defaults to ``None``.
        :type chain_selections1: Optional[List[str]], optional
        :param graphein_config: Protein graph construction config, defaults to
            ``ProteinGraphConfig()``.
        :type graphein_config: ProteinGraphConfig, optional
        :param graph_format_convertor: Conversion handler for graphs, defaults
            to ``GraphFormatConvertor(src_format="nx", dst_format="pyg")``.
        :type graph_format_convertor: GraphFormatConvertor, optional
        :param graph_transformation_funcs: List of functions that consume a
            ``nx.Graph`` and return a ``nx.Graph``. Applied to graphs after
            construction but before conversion to pyg. Defaults to ``None``.
        :type graph_transformation_funcs: Optional[List[Callable]], optional
        :param pdb_transform: List of functions that consume a list of paths to
            the downloaded structures. This provides an entry point to apply
            pre-processing from bioinformatics tools of your choosing. Defaults
            to ``None``.
        :type pdb_transform: Optional[List[Callable]], optional
        :param paired_transform: A function/transform that takes in a
            ``torch_geometric.data.Data`` object and returns a transformed
            version. The data object will be transformed before every access.
            Defaults to ``None``. It is applied after the paired data have been generated.
        :type paired_transform: Optional[Callable], optional
        :param paired_pre_transform:  A function/transform that takes in an
            ``torch_geometric.data.Data`` object and returns a transformed
            version. The data object will be transformed before being saved to
            disk. Defaults to ``None``. It is applied after the paired data have been generated.
        :type paired_pre_transform: Optional[Callable], optional
        :param paired_pre_filter:  A function that takes in a
            ``torch_geometric.data.Data`` object and returns a boolean value,
            indicating whether the data object should be included in the final
            dataset. Optional, defaults to ``None``. It is applied after the paired data have been generated.
        :type paired_pre_filter: Optional[Callable], optional
        :param transform: A function/transform that takes in a
            ``torch_geometric.data.Data`` object and returns a transformed
            version. The data object will be transformed before every access.
            Defaults to ``None``.
        :type transform: Optional[Callable], optional
        :param pre_transform:  A function/transform that takes in an
            ``torch_geometric.data.Data`` object and returns a transformed
            version. The data object will be transformed before being saved to
            disk. Defaults to ``None``.
        :type pre_transform: Optional[Callable], optional
        :param pre_filter:  A function that takes in a
            ``torch_geometric.data.Data`` object and returns a boolean value,
            indicating whether the data object should be included in the final
            dataset. Optional, defaults to ``None``.
        :type pre_filter: Optional[Callable], optional
        :param num_cores: Number of cores to use for multiprocessing of graph
            construction, defaults to ``16``.
        :type num_cores: int, optional
        :param af_version: Version of AlphaFoldDB structures to use,
            defaults to ``2``.
        :type af_version: int, optional
        """

        flag: bool = False
        if pdb_paths0 is not None and pdb_paths1 is not None:
            flag = True
            if len(pdb_codes0) != len(pdb_codes1):
                raise ValueError(f"pdb_paths0 and pdb_paths1 should have the same length, got {len(pdb_paths0)} and "
                                 f"{len(pdb_paths1)}")
        if pdb_codes0 is not None and pdb_codes1 is not None:
            if flag:
                raise ValueError("Only one parameter between pdb_paths/pdb_codes/uniprot_ids should be given, 2 given.")
            flag = True
            if len(pdb_codes0) != len(pdb_codes1):
                raise ValueError(f"pdb_codes0 and pdb_codes1 should have the same length, got {len(pdb_codes0)} and "
                                 f"{len(pdb_codes1)}")
        if uniprot_ids0 is not None and uniprot_ids1 is not None:
            if flag:
                raise ValueError("Only one parameter between pdb_paths/pdb_codes/uniprot_ids should be given, 2 given.")
            flag = True
            if len(pdb_codes0) != len(pdb_codes1):
                raise ValueError(f"uniprot_ids0 and uniprot_ids1 should have the same length, got {len(uniprot_ids0)} "
                                 f"and {len(uniprot_ids1)}")
        if not flag:
            raise ValueError("Exactly one couple of parameters type should be given. 0 given.")

        if len(graph_labels0) != len(graph_labels1):
            raise ValueError(f"graph_labels0 and graph_labels1 must have the same length. {len(graph_labels0)} and "
                             f"{len(graph_labels1)} given.")

        self.__dataset0 = ProteinGraphDataset(
            root=root,
            pdb_paths=pdb_paths0,
            pdb_codes=pdb_codes0,
            uniprot_ids=uniprot_ids0,
            graph_labels=graph_labels0,
            node_labels=node_labels0,
            chain_selections=chain_selections0,
            graphein_config=graphein_config,
            graph_format_convertor=graph_format_convertor,
            graph_transformation_funcs=graph_transformation_funcs,
            pdb_transform=pdb_transform,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            num_cores=num_cores,
            af_version=af_version
        )
        self.__dataset1 = ProteinGraphDataset(
            root=root,
            pdb_paths=pdb_paths1,
            pdb_codes=pdb_codes1,
            uniprot_ids=uniprot_ids1,
            graph_labels=graph_labels1,
            node_labels=node_labels1,
            chain_selections=chain_selections1,
            graphein_config=graphein_config,
            graph_format_convertor=graph_format_convertor,
            graph_transformation_funcs=graph_transformation_funcs,
            pdb_transform=pdb_transform,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            num_cores=num_cores,
            af_version=af_version
        )
        super().__init__(
            root,
            transform=paired_transform,
            pre_transform=paired_pre_transform,
            pre_filter=paired_pre_filter
        )

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files in the dataset."""
        raw_file_names: list[str] = []
        for pdb0, pdb1 in zip(self.__dataset0.raw_file_names, self.__dataset1.raw_file_names):
            raw_file_names.extend([pdb0, pdb1])
        return raw_file_names

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files to look for"""
        processed_file_names: list[str] = []
        for pyg0, pyg1 in zip(self.__dataset0.processed_file_names, self.__dataset1.processed_file_names):
            processed_file_names.extend([pyg0, pyg1])
        return processed_file_names

    @property
    def raw_dir(self) -> str:
        return self.__dataset0.raw_dir  # should be self.__dataset1.raw_dir also taken into account?

    def len(self) -> int:
        """Returns length of data set (number of structures)."""
        return len(self.__dataset0)

    def process(self):
        pass

    def download(self):
        pass

    def get(self, idx: int) -> Data:
        """
       Gets the i-th (before, after) protein graph pair..

        :param idx: Index to retrieve.
        :type idx: int
        :return: PyTorch Geometric Data object.
        """
        before = self.__dataset0.get(idx)
        after = self.__dataset1.get(idx)

        return PairedData(before, after, global_y=before.graph_y)


class PairedCollater(Collater):
    def __init__(self, follow_batch: Optional[List[str]], exclude_keys: Optional[List[str]]):
        """
        A collater to create batch in torch datasets containing pairs of graphs.

        :param follow_batch: follow_batch: creates assignment batch vectors for each key in the list (default:
            :obj:`None`)
        :type follow_batch: Optional[List[str]]
        :param exclude_keys: will exclude each key in the list (default: :obj:`None`)
        :type exclude_keys: Optional[List[str]]
        """
        super(PairedCollater, self).__init__(follow_batch=follow_batch, exclude_keys=exclude_keys)

    def __call__(self, batch):
        """
        Collates the given batch of graph pairs.

        :param batch: batch to collate as an iterable

        :return: the collated batch of paired graphs.
        """

        # Create a batch which contains a pair of lists of Data objects
        data_list_batch = super(PairedCollater, self).__call__(batch)

        # Get the two data lists and convert them into batches
        data_list_a = data_list_batch.a
        data_list_b = data_list_batch.b
        batch_a = Batch.from_data_list(data_list_a)
        batch_b = Batch.from_data_list(data_list_b)

        # Create the PairedData which represents the paired batch
        paired_batch = PairedData(a=batch_a, b=batch_b, global_y=data_list_batch.y)

        return paired_batch


class PairedDataLoader(DataLoader):
    def __init__(
            self,
            dataset: Union[Dataset, List[BaseData]],
            batch_size: int = 1,
            shuffle: bool = True,
            follow_batch: Optional[List[str]] = None,
            exclude_keys: Optional[List[str]] = None,
            **kwargs,
    ):
        """
        DataLoader to load batches of pairs of graphs. Uses fixed custom collate function from PairedCollater class.

        :param dataset: the dataset to load the pairs of graphs from
        :type dataset: Union[Dataset, List[BaseData]]
        :param batch_size: how many samples per batch to load (default: 1)
        :type batch_size: int
        :param shuffle: if set to :obj:`True`, the data will be reshuffled at every epoch (default: :obj:`False`)
        :type shuffle: boolean
        :param follow_batch: follow_batch: creates assignment batch vectors for each key in the list (default:
            :obj:`None`)
        :type follow_batch: Optional[List[str]]
        :param exclude_keys: will exclude each key in the list (default: :obj:`None`)
        :type exclude_keys: Optional[List[str]]
        """

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=PairedCollater(follow_batch=follow_batch, exclude_keys=exclude_keys),
            **kwargs,
        )
