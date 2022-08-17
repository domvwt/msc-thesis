"""
Load graph data from parquet files into PyTorch Geometric format.

Based on the CSV loader example at https://pytorch-geometric.readthedocs.io/en/latest/notes/load_csv.html.
"""

from typing import Iterable, Tuple, List

import pandas as pd
import torch
from torch_geometric.data import HeteroData

import pdcast as pdc


def node_df_to_pyg(
    df: pd.DataFrame, id_col, mapping_start, encoders, label_col, **kwargs
):
    mapping = {
        str(node_id): idx + mapping_start
        for idx, node_id in enumerate(df[id_col].unique())
    }

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    y = None
    if label_col is not None:
        y = torch.tensor(df[label_col].astype(int).values)

    return x, y, mapping


def edge_df_to_pyg(
    df: pd.DataFrame, src_col, dst_col, id_idx_mapping, encoders=None, **kwargs
):
    if df.empty:
        return torch.tensor([]), torch.tensor([])

    src, dst = zip(
        *(
            (id_idx_mapping[src_index], id_idx_mapping[dst_index])
            for src_index, dst_index in zip(df[src_col], df[dst_col])
        )
    )

    edge_index = torch.tensor([src, dst])
    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


class IdentityEncoder(object):
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)


def get_processed_feature_cols(df: pd.DataFrame) -> Iterable[str]:
    """
    Get the list of feature columns that have been processed.
    """
    return [col for col in df.columns if col.endswith("__processed")]


def make_identity_encoders(cols):
    return {col: IdentityEncoder() for col in cols}


def load_data_to_pyg(
    companies_path, persons_path, edges_path
) -> List[HeteroData]:
    companies_df = pd.read_parquet(companies_path)
    persons_df = pd.read_parquet(persons_path)
    edges_df = pd.read_parquet(edges_path)

    all_edge_ids = pd.concat((edges_df["src"], (edges_df["dst"])), axis=0).unique()
    missing_companies = companies_df.query("id not in @all_edge_ids")
    missing_persons = persons_df.query("id not in @all_edge_ids")
    if len(missing_companies) > 0 or len(missing_persons) > 0:
        raise ValueError("Missing companies or persons")

    # Downcast to numpy dtypes.
    companies_df: pd.DataFrame = pdc.downcast(companies_df, numpy_dtypes_only=True)  # type: ignore
    persons_df: pd.DataFrame = pdc.downcast(persons_df, numpy_dtypes_only=True)  # type: ignore
    edges_df: pd.DataFrame = pdc.downcast(edges_df, numpy_dtypes_only=True)  # type: ignore

    # Split into separate batches as PyG expects.
    data_list = []

    companies_features = get_processed_feature_cols(companies_df)
    persons_features = get_processed_feature_cols(persons_df)
    edges_features = get_processed_feature_cols(edges_df)

    companies_encoders = make_identity_encoders(companies_features)
    persons_encoders = make_identity_encoders(persons_features)
    edges_encoders = make_identity_encoders(edges_features)

    for component_id in companies_df["component"].unique():

        component_id: int

        component_companies_df = companies_df.query("component == @component_id")
        component_persons_df = persons_df.query("component == @component_id")
        component_edges_df = edges_df.query("component == @component_id")

        companies_x, companies_y, companies_mapping = node_df_to_pyg(
            component_companies_df,
            "id",
            0,
            encoders=companies_encoders,
            label_col="is_anomalous",
        )

        persons_x, persons_y, persons_mapping = node_df_to_pyg(
            component_persons_df,
            "id",
            len(companies_mapping),
            encoders=persons_encoders,
            label_col="is_anomalous",
        )

        company_company_edges = component_edges_df.query(
            "interestedPartyIsPerson == False"
        )
        person_company_edges = component_edges_df.query(
            "interestedPartyIsPerson == True"
        )

        combined_mapping = {**companies_mapping, **persons_mapping}

        company_company_edge_index, company_company_edge_attr = edge_df_to_pyg(
            company_company_edges,
            "src",
            "dst",
            combined_mapping,
            encoders=edges_encoders,
        )

        person_company_edge_index, person_company_edge_attr = edge_df_to_pyg(
            person_company_edges,
            "src",
            "dst",
            combined_mapping,
            encoders=edges_encoders,
        )

        pyg_data = HeteroData()
        pyg_data["company"].x = companies_x  # type: ignore
        pyg_data["company"].y = companies_y  # type: ignore
        pyg_data["person"].x = persons_x  # type: ignore
        pyg_data["person"].y = persons_y  # type: ignore
        pyg_data["company", "owns", "company"].edge_index = company_company_edge_index  # type: ignore
        pyg_data["company", "owns", "company"].edge_attr = company_company_edge_attr  # type: ignore
        pyg_data["person", "owns", "company"].edge_index = person_company_edge_index  # type: ignore
        pyg_data["person", "owns", "company"].edge_attr = person_company_edge_attr  # type: ignore

        # data: Tuple[int, HeteroData] = (component_id, pyg_data)
        data_list.append(pyg_data)

    return data_list
