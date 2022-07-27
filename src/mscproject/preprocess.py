import dataclasses as dc

import numpy as np
import pandas as pd
from sklearn import pipeline
import sklearn.preprocessing as pre
import yaml

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

import pdcast as pdc

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

    def get_feature_names_out(self, input_features=None):
        return self.columns


class DateConverter(BaseEstimator, TransformerMixin):
    def __init__(self, date_cols):
        self.date_cols = date_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.date_cols:
            X.loc[:, col] = pd.to_datetime(X[col])
            X.loc[:, col] = X[col].astype(np.int64).to_numpy()
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


class Downcaster(BaseEstimator, TransformerMixin):
    def __init__(self, numpy_dtypes_only: bool = True):
        self.numpy_dtypes_only = numpy_dtypes_only

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pdc.downcast(X, numpy_dtypes_only=self.numpy_dtypes_only)

    def get_feature_names_out(self, input_features=None):
        return input_features


def build_pipeline(
    keep_cols: list,
    date_cols: list,
    one_hot_encoder_kwargs: dict,
    numpy_dtypes_only: bool = True,
) -> Pipeline:
    return make_pipeline(
        ColumnSelector(keep_cols),
        DateConverter(date_cols),
        Downcaster(numpy_dtypes_only=numpy_dtypes_only),
        make_column_transformer(
            (
                OneHotEncoder(**one_hot_encoder_kwargs),
                make_column_selector(dtype_include=object),
            ),
            remainder="passthrough",

        ),
        StandardScaler(),
    )


def apply_pipeline(
    df: pd.DataFrame, pipeline: Pipeline, meta_cols: list[str]
) -> pd.DataFrame:
    df_out = pd.DataFrame(
        pipeline.transform(df), columns=pipeline.get_feature_names_out()
    )
    # Remove leading 'remainder__' from column names
    df_out.columns = [
        col.split("remainder__")[-1] for col in df_out.columns
    ]
    return df[meta_cols].join(df_out)


def process_companies(train_df, valid_df, test_df):

    feature_cols = [
        # "id",
        # "component",
        # "isCompany",
        # "name",
        "foundingDate",
        # "dissolutionDate",  # ! practically invariant
        "countryCode",
        # "companiesHouseID",
        # "openCorporatesID",
        # "openOwnershipRegisterID",
        # "CompanyCategory",  # ! practically invariant
        "CompanyStatus",
        "Accounts_AccountCategory",
        "SICCode_SicText_1",
        "indegree",
        "outdegree",
        "closeness",
        "clustering",
        "pagerank",
        "neighbourhood_indegree",
        "neighbourhood_outdegree",
        "neighbourhood_closeness",
        "neighbourhood_clustering",
        "neighbourhood_pagerank",
        "neighbourhood_num_neighbours",
        "is_anomalous",
    ]

    date_cols = ["foundingDate"]

    one_hot_encoder_kwargs = dict(
        drop="first",
        sparse=False,
        dtype=np.uint8,
        handle_unknown="infrequent_if_exist",
        min_frequency=1000,
    )

    meta_cols = ["id", "component"]
    pipeline = build_pipeline(feature_cols, date_cols, one_hot_encoder_kwargs)

    pipeline.fit(train_df)
    train_df = apply_pipeline(train_df, pipeline, meta_cols)
    valid_df = apply_pipeline(valid_df, pipeline, meta_cols)
    test_df = apply_pipeline(test_df, pipeline, meta_cols)

    return train_df, valid_df, test_df


def process_persons(train_df, valid_df, test_df):

    feature_cols = [
        # "id",
        # "component",
        # "isCompany",
        "birthDate",
        # "name",
        "nationality",
        "indegree",
        "outdegree",
        "closeness",
        "clustering",
        "pagerank",
        "neighbourhood_indegree",
        "neighbourhood_outdegree",
        "neighbourhood_closeness",
        "neighbourhood_clustering",
        "neighbourhood_pagerank",
        "neighbourhood_num_neighbours",
        "is_anomalous",
    ]

    date_cols = ["birthDate"]

    one_hot_encoder_kwargs = dict(
        drop="first",
        sparse=False,
        dtype=np.uint8,
        handle_unknown="infrequent_if_exist",
        min_frequency=50,
    )

    meta_cols = ["id", "component"]
    pipeline = build_pipeline(feature_cols, date_cols, one_hot_encoder_kwargs)

    pipeline.fit(train_df)
    train_df = apply_pipeline(train_df, pipeline, meta_cols)
    valid_df = apply_pipeline(valid_df, pipeline, meta_cols)
    test_df = apply_pipeline(test_df, pipeline, meta_cols)

    return train_df, valid_df, test_df


def process_edges(train_df, valid_df, test_df):

    keep_cols = [
        # "component",
        # "src",
        # "dst",
        # "interestedPartyIsPerson",
        "minimumShare",
        # "is_anomalous",
    ]

    meta_cols = ["src", "dst", "component"]
    pipeline = build_pipeline(keep_cols, [], {})

    pipeline.fit(train_df)
    train_df = apply_pipeline(train_df, pipeline, meta_cols)
    valid_df = apply_pipeline(valid_df, pipeline, meta_cols)
    test_df = apply_pipeline(test_df, pipeline, meta_cols)

    return train_df, valid_df, test_df


def split_datasets(companies_df, persons_df, edges_df):
    components = persons_df.component.unique()
    component_assignment_df = pd.DataFrame(
        np.array([components, components % 10]).T,
        columns=["component", "component_mod"],
    ).sort_values(by="component")

    train_idx = component_assignment_df.query(
        "component_mod >= 1 and component_mod <= 8"
    ).index
    valid_idx = component_assignment_df.query("component_mod >= 9").index
    test_idx = component_assignment_df.query("component_mod == 0").index

    component_assignment_df.loc[train_idx, "split"] = "train"
    component_assignment_df.loc[valid_idx, "split"] = "valid"
    component_assignment_df.loc[test_idx, "split"] = "test"

    # Used in the following queries.
    train_components = component_assignment_df.query("split == 'train'")[
        "component"
    ].to_list()
    valid_components = component_assignment_df.query("split == 'valid'")[
        "component"
    ].to_list()
    test_components = component_assignment_df.query("split == 'test'")[
        "component"
    ].to_list()

    train_person_nodes = persons_df.query("component in @train_components")
    valid_person_nodes = persons_df.query("component in @valid_components")
    test_person_nodes = persons_df.query("component in @test_components")

    train_company_nodes = companies_df.query("component in @train_components")
    valid_company_nodes = companies_df.query("component in @valid_components")
    test_company_nodes = companies_df.query("component in @test_components")

    train_edges = edges_df.query("src in @train_person_nodes.id")
    valid_edges = edges_df.query("src in @valid_person_nodes.id")
    test_edges = edges_df.query("src in @test_person_nodes.id")

    return (
        train_company_nodes,
        train_person_nodes,
        train_edges,
        valid_company_nodes,
        valid_person_nodes,
        valid_edges,
        test_company_nodes,
        test_person_nodes,
        test_edges,
    )
