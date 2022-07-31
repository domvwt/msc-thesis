import dataclasses as dc
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdcast as pdc
import sklearn.preprocessing as pre
import yaml
from sklearn import pipeline
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


def apply_pipeline(df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    df_out = pd.DataFrame(
        pipeline.transform(df), columns=pipeline.get_feature_names_out()
    )
    # Remove leading 'remainder__' from column names
    df_out.columns = [
        col.split("remainder__")[-1] + "__processed" for col in df_out.columns
    ]
    return df.join(df_out)


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
        "neighbourhood_neighbour_count",
        # "is_anomalous",
    ]

    date_cols = ["foundingDate"]

    one_hot_encoder_kwargs = dict(
        drop="first",
        sparse=False,
        dtype=np.uint8,
        handle_unknown="infrequent_if_exist",
        min_frequency=1000,
    )

    pipeline = build_pipeline(feature_cols, date_cols, one_hot_encoder_kwargs)

    pipeline.fit(train_df)
    train_df = apply_pipeline(train_df, pipeline)
    valid_df = apply_pipeline(valid_df, pipeline)
    test_df = apply_pipeline(test_df, pipeline)

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
        "neighbourhood_neighbour_count",
        # "is_anomalous",
    ]

    date_cols = ["birthDate"]

    one_hot_encoder_kwargs = dict(
        drop="first",
        sparse=False,
        dtype=np.uint8,
        handle_unknown="infrequent_if_exist",
        min_frequency=50,
    )

    pipeline = build_pipeline(feature_cols, date_cols, one_hot_encoder_kwargs)

    pipeline.fit(train_df)
    train_df = apply_pipeline(train_df, pipeline)
    valid_df = apply_pipeline(valid_df, pipeline)
    test_df = apply_pipeline(test_df, pipeline)

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

    pipeline = build_pipeline(keep_cols, [], {})

    pipeline.fit(train_df)
    train_df = apply_pipeline(train_df, pipeline)
    valid_df = apply_pipeline(valid_df, pipeline)
    test_df = apply_pipeline(test_df, pipeline)

    return train_df, valid_df, test_df


def split_dataset(df) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    component_mod = (df.component % 10).to_numpy()

    train_df = df.loc[component_mod <= 6, :]
    valid_df = df.loc[(7 <= component_mod) & (component_mod <= 8)]
    test_df = df.loc[component_mod == 9]

    return train_df, valid_df, test_df
