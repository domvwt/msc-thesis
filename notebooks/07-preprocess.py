# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: 'Python 3.9.12 (''.venv'': poetry)'
#     language: python
#     name: python3
# ---

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

import pdcast as pdc

while not Path("data") in Path(".").iterdir():
    os.chdir("..")

import sklearn.preprocessing as pre


# %%
# Read config.
conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

persons_df = pd.read_parquet(conf_dict["persons_nodes"])
companies_df = pd.read_parquet(conf_dict["companies_nodes"])
edges_df = pd.read_parquet(conf_dict["edges"])


# %%
# function to convert pandas columns to datetime
def convert_to_datetime(df, cols):
    for col in cols:
        df[col] = pd.to_datetime(df[col])
    return df



# %%
persons_df = convert_to_datetime(persons_df, ["birthDate"])

# %%
companies_df = convert_to_datetime(companies_df, ["foundingDate", "dissolutionDate"])

# %%
company_drop_cols = [
    "dissolutionDate", # practically invariant
    "CompanyCategory", # practically invariant
]

# %%
persons_one_hot_encoder_kwargs = dict(
    drop="first",
    sparse=False,
    dtype=np.uint8,
    handle_unknown="infrequent_if_exist",
    min_frequency=50,
)

companies_one_hot_encoder_kwargs = dict(
    drop="first",
    sparse=False,
    dtype=np.uint8,
    handle_unknown="infrequent_if_exist",
    min_frequency=1000,
)


# %%
persons_df.info()

# %%
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer


# %%
def identity(x):
    return x

passthough_kwargs = dict(
    func=identity,
    feature_names_out="one-to-one"
)

# %%
companies_ct = ColumnTransformer(
    [
        (
            "one_hot_encoder",
            pre.OneHotEncoder(**companies_one_hot_encoder_kwargs),
            [
                "countryCode",
                "CompanyStatus",
                "Accounts_AccountCategory",
                "SICCode_SicText_1",
            ],
        ),
        ("standard_scaler", pre.StandardScaler(), ["foundingDate"]),
        # pass through the id and component
        # ("passthrough", pre.FunctionTransformer(**passthough_kwargs), ["id", "component"]),
    ],
    remainder="drop",
)


# %%
persons_ct = ColumnTransformer(
    [
        ("one_hot_encoder", pre.OneHotEncoder(**persons_one_hot_encoder_kwargs), ["nationality"]),
        ("standard_scaler", pre.StandardScaler(), ["birthDate"]),
        # pass through the id and component
        # ("passthrough", pre.FunctionTransformer(**passthough_kwargs), ["id", "component"]),
    ],
    remainder="drop",
)


# %%
def remove_from_col_names(df, s):
    df.columns = [col.replace(s, "") for col in df.columns]
    return df


# %%
persons_processed_df = persons_ct.fit_transform(persons_df)
persons_processed_df = pd.DataFrame(persons_processed_df, columns=persons_ct.get_feature_names_out())
persons_processed_df = pdc.downcast(persons_processed_df, numpy_dtypes_only=True)
persons_processed_df = persons_df[["id", "component"]].join(persons_processed_df)
persons_processed_df.info()

# %%
companies_processed_df = companies_ct.fit_transform(companies_df)
companies_processed_df = pd.DataFrame(companies_processed_df, columns=companies_ct.get_feature_names_out())
companies_processed_df = remove_from_col_names(companies_processed_df, "passthrough__")
companies_processed_df = pdc.downcast(companies_processed_df, numpy_dtypes_only=True)
companies_processed_df = companies_df[["id", "component"]].join(companies_processed_df)
companies_processed_df.info()

# %%
edges_processed_df = edges_df.drop(columns=["interestedPartyIsPerson"])
edges_processed_df["minimumShare"] = pre.StandardScaler().fit_transform(edges_processed_df["minimumShare"].values.reshape(-1, 1))

# %%
edges_processed_df
