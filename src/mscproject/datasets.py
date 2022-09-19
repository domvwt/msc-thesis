from pathlib import Path
from typing import Callable, Optional

import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.transforms.to_undirected import ToUndirected


class CompanyBeneficialOwners(InMemoryDataset):
    """
    A PyTorch Geometric dataset for UK companies and their Beneficial Owners.
    https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html?highlight=inmemorydataset#creating-in-memory-datasets
    """

    data: HeteroData

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        to_undirected: bool = False,
    ):
        self.name = type(self).__name__
        super().__init__(root, transform, pre_transform)
        loaded_data = torch.load(self.processed_paths[0])
        if to_undirected:
            loaded_data = ToUndirected(merge=False)(loaded_data)
        self.data, self.slices = self.collate([loaded_data])  # type: ignore

    @property
    def processed_dir(self):
        assert self.root, "Please specify a root directory"
        return str(Path(self.root) / "processed")

    @property
    def processed_file_names(self):
        assert self.processed_dir is not None, "Please specify `processed_dir`"
        return "data.pt"

    def metadata(self):
        return self.data.metadata()
