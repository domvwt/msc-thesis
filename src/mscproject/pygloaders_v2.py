"""
Load graph data from parquet files into PyTorch Geometric format.

Based on the CSV loader example at https://pytorch-geometric.readthedocs.io/en/latest/notes/load_csv.html.
"""

from collections import namedtuple
from typing import Iterable

import joblib as jl
import pandas as pd
import pdcast as pdc
import torch
from tqdm.auto import tqdm
from torch_geometric.data import HeteroData


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


Masks = namedtuple("Masks", ["train", "val", "test"])


def get_data_split_masks(df: pd.DataFrame) -> Masks:
    component_mod = df["component"].to_numpy() % 10
    train_mask = torch.tensor(component_mod <= 7)
    val_mask = torch.tensor(component_mod == 8)
    test_mask = torch.tensor(component_mod == 9)
    return Masks(train_mask, val_mask, test_mask)

def masks_unique(masks: Masks) -> bool:
    combined = masks.train.int() + masks.val.int() + masks.test.int()
    return combined.sum() == len(combined)

def graph_elements_to_heterodata(
    companies_df: pd.DataFrame, persons_df: pd.DataFrame, edges_df: pd.DataFrame
) -> list[HeteroData]:

    component_labels = companies_df["component"].unique()

    all_edge_ids = pd.concat((edges_df["src"], (edges_df["dst"])), axis=0).unique()
    missing_companies = companies_df.query("id not in @all_edge_ids")
    missing_persons = persons_df.query("id not in @all_edge_ids")
    if len(missing_companies) > 0 or len(missing_persons) > 0:
        raise ValueError("Missing companies or persons")

    pyg_data_list = []
    append = pyg_data_list.append

    def process(component_label):
        component_companies_df = companies_df.query("component == @component_label")
        component_persons_df = persons_df.query("component == @component_label")
        component_edges_df = edges_df.query("component == @component_label")

        # Downcast to numpy dtypes.
        component_companies_df: pd.DataFrame = pdc.downcast(component_companies_df, numpy_dtypes_only=True)  # type: ignore
        component_persons_df: pd.DataFrame = pdc.downcast(component_persons_df, numpy_dtypes_only=True)  # type: ignore
        component_edges_df: pd.DataFrame = pdc.downcast(component_edges_df, numpy_dtypes_only=True)  # type: ignore

        assert component_edges_df.shape[0] >= component_companies_df.shape[0] + component_persons_df.shape[0] - 1

        companies_features = get_processed_feature_cols(component_companies_df)
        persons_features = get_processed_feature_cols(component_persons_df)
        edges_features = get_processed_feature_cols(component_edges_df)

        companies_encoders = make_identity_encoders(companies_features)
        persons_encoders = make_identity_encoders(persons_features)
        edges_encoders = make_identity_encoders(edges_features)

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
            0,
            encoders=persons_encoders,
            label_col="is_anomalous",
        )

        component_edges_df["interestedPartyIsPerson"] = component_edges_df["src"].isin(persons_mapping)
        company_company_edges = component_edges_df.query("interestedPartyIsPerson == False")
        person_company_edges = component_edges_df.query("interestedPartyIsPerson == True")

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

        return pyg_data

    # pyg_data_list = jl.Parallel(n_jobs=8)(jl.delayed(process)(component_label) for component_label in tqdm(component_labels))
    pyg_data_list = [process(component_label) for component_label in tqdm(component_labels)]

    return pyg_data_list
