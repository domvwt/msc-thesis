"""Load graph data from parquet files into PyTorch Geometric format."""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import yaml

from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit, ToUndirected

def load_edge_parquet(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_parquet(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr

def load_node_parquet(path, index_col, mapping_start, encoders=None, **kwargs):
    df = pd.read_parquet(path, **kwargs).set_index(index_col)
    mapping = {index: i + mapping_start for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping

class GenresEncoder(object):
    # The 'GenreEncoder' splits the raw column strings by 'sep' and converts
    # individual elements to categorical labels.
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        genres = set(g for col in df.values for g in col.split(self.sep))
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x

class IdentityEncoder(object):
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)


def main():

    # Read config.
    conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

    edges_path = conf_dict["persons_nodes"]
    persons_path = conf_dict["companies_nodes"]
    edges_path = conf_dict["edges"]
    connected_components_path = conf_dict["connected_components"]

    companies_encoders = {
        # 'id',
        # 'component',
        # 'isCompany',
        # 'name',
        'foundingDate': None,
        'dissolutionDate': None,
        'countryCode': None,
        # 'companiesHouseID',
        # 'openCorporatesID',
        # 'openOwnershipRegisterID',
        'CompanyCategory': None,
        'CompanyStatus': None,
        'Accounts_AccountCategory': None,
        'SICCode_SicText_1': None
    }

    persons_encoders = {
        # 'id', 
        # 'component', 
        # 'isCompany', 
        'birthDate': None, 
        # 'name', 
        'nationality': None
    }

    edge_encoder = {
        'minimumShare': None
    }

    companies_x, companies_mapping = load_node_parquet(edges_path, index_col='id', mapping_start=0)
    persons_x, persons_mapping = load_node_parquet(persons_path, index_col='id', mapping_start=len(companies_mapping))
    combined_mapping = {**companies_mapping, **persons_mapping}

    # raise Exception("Done!")

    persons_x, persons_mapping = load_node_parquet(
        persons_path, index_col='movieId', encoders={
            'title': SequenceEncoder(),
            'genres': GenresEncoder()
        })

    edge_index, edge_label = load_edge_parquet(
        edges_path,
        src_index_col='src',
        src_mapping=combined_mapping,
        dst_index_col='dst',
        dst_mapping=combined_mapping,
        encoders={'rating': IdentityEncoder(dtype=torch.long)},
    )

    data = HeteroData()
    data['user'].x = companies_x
    data['movie'].x = persons_x
    data['user', 'rates', 'movie'].edge_index = edge_index
    data['user', 'rates', 'movie'].edge_label = edge_label
    print(data)

    # We can now convert `data` into an appropriate format for training a
    # graph-based machine learning model:

    # 1. Add a reverse ('movie', 'rev_rates', 'user') relation for message passing.
    data = ToUndirected()(data)
    del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.

    # 2. Perform a link-level split into training, validation, and test edges.
    transform = RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        neg_sampling_ratio=0.0,
        edge_types=[('user', 'rates', 'movie')],
        rev_edge_types=[('movie', 'rev_rates', 'user')],
    )
    train_data, val_data, test_data = transform(data)
    print(train_data)
    print(val_data)
    print(test_data)

if __name__ == "__main__":
    main()
