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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
import yaml

import pandas_profiling as pp

import graphframes as gf
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Column

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
persons_df.groupby("component")["id"].count().sort_values(ascending=False)

# %%
# select 10% of the person nodes
anomalous_persons_df = persons_df.sample(frac=0.1, random_state=42)

# %%
# select 10% of the company nodes
anomalous_companies_df = companies_df.sample(frac=0.1, random_state=42)

# %%
# flag the anomalous entities
persons_df["is_anomalous"] = False
persons_df.loc[anomalous_persons_df.index, "is_anomalous"] = True
companies_df["is_anomalous"] = False
companies_df.loc[anomalous_companies_df.index, "is_anomalous"] = True


# %%
# select edges for anomalous persons and companies
# anomalous_edges_df = edges_df[edges_df["src"].isin(anomalous_persons_df["id"])]
anomalous_edges_df = edges_df[
    edges_df["src"].isin(
        set(
            anomalous_persons_df["id"].to_list()
            + anomalous_companies_df["id"].to_list()
        )
    )
]
anomalous_edges_df = anomalous_edges_df.copy(deep=True)


# %%

# select array indexes where values are equal
def indexes_not_shuffled(a1, a2):
    return np.argwhere(a1 == a2)


def efficient_shuffle(a1):
    def inner(a1, _i):
        print(_i, end="\r")
        a2 = a1.copy()
        rng = np.random.default_rng(42 + _i)
        for i in range(5):
            to_shuffle = indexes_not_shuffled(a1, a2)
            a2[to_shuffle] = rng.permutation(a2[to_shuffle])
            if all_shuffled(a1, a2):
                break
        else:
            inner(a1, _i + 1)
        return a2
    return inner(a1, 0)



# %%
# permute the edges from anomalous entities until they are all shuffled

rng = np.random.default_rng(42)

shuffled_edges_df = anomalous_edges_df.copy()

def all_shuffled(a1, a2):
    return np.all(a1 != a2)

i = 0

original_edges = shuffled_edges_df["src"].to_numpy()

shuffled_edges = efficient_shuffle(original_edges)

shuffled_edges_df["src"] = shuffled_edges


# %%
# replace the edges with the shuffled ones
edges_anomalised_df = edges_df.copy(deep=True)
edges_anomalised_df = edges_anomalised_df.drop(shuffled_edges_df.index)
edges_anomalised_df = pd.concat([edges_anomalised_df, shuffled_edges_df]).sort_index()
edges_anomalised_df["is_anomalous"] = False
edges_anomalised_df.loc[anomalous_edges_df.index, "is_anomalous"] = True

# %%
assert len(edges_anomalised_df) == len(edges_df)
assert not edges_anomalised_df.equals(edges_df)

# %%
persons_df.groupby("is_anomalous")["id"].count().sort_values(ascending=False)

# %%
companies_df.groupby("is_anomalous")["id"].count().sort_values(ascending=False)

# %%
edges_anomalised_df.groupby("is_anomalous")["src"].count().sort_values(ascending=False)

# %%
components = persons_df.component.unique()

# %%
component_assignment_df = pd.DataFrame(
    np.array([components, components % 10]).T, columns=["component", "component_mod"]
).sort_values(by="component")

train_idx = component_assignment_df.query("component_mod >= 1 and component_mod <= 8").index
valid_idx = component_assignment_df.query("component_mod >= 9").index
test_idx = component_assignment_df.query("component_mod == 0").index

component_assignment_df.loc[train_idx, "split"] = "train"
component_assignment_df.loc[valid_idx, "split"] = "valid"
component_assignment_df.loc[test_idx, "split"] = "test"


# %%
component_assignment_df.groupby("split")["component"].count().sort_values(ascending=False)

# %%
train_components = component_assignment_df.query("split == 'train'")["component"].to_list()
valid_components = component_assignment_df.query("split == 'valid'")["component"].to_list()
test_components = component_assignment_df.query("split == 'test'")["component"].to_list()

# %%
train_person_nodes = persons_df.query("component in @train_components")
valid_person_nodes = persons_df.query("component in @valid_components")
test_person_nodes = persons_df.query("component in @test_components")

train_company_nodes = companies_df.query("component in @train_components")
valid_company_nodes = companies_df.query("component in @valid_components")
test_company_nodes = companies_df.query("component in @test_components")

train_edges = edges_anomalised_df.query("src in @train_person_nodes.id")
valid_edges = edges_anomalised_df.query("src in @valid_person_nodes.id")
test_edges = edges_anomalised_df.query("src in @test_person_nodes.id")

# %%
