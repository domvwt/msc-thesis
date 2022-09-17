# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
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
import torch
import sklearn
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import yaml

import mscproject.models as mod
import mscproject.experiment as exp

while not Path("data") in Path(".").iterdir():
    os.chdir("..")

# %%
# Matplotlib Settings
plt.style.use("seaborn-whitegrid")
plt.style.use("seaborn-paper")

FONT_SIZE = 12

# Set plot font size
plt.rcParams.update({"font.size": 8})

# Set axis label font size
plt.rcParams.update({"axes.labelsize": 10})

# Set legend font size
plt.rcParams.update({"legend.fontsize": 10})

# Set tick label font size
plt.rcParams.update({"xtick.labelsize": 9})

# Set tick label font size
plt.rcParams.update({"ytick.labelsize": 9})

# Set figure title font size
plt.rcParams.update({"axes.titlesize": FONT_SIZE})

# %%
conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

# %%
edges_pre_df = pd.read_parquet(conf_dict["edges"])

# %%
edges_post_df = pd.read_parquet(conf_dict["edges_anomalies"])

# %%

# %% [markdown]
# ## Graph Statistics

# %%
edges_post_df.is_anomalous.value_counts(normalize=True)

# %%
# Get component sizes
component_sizes_pre = edges_pre_df.groupby("component").agg({"src": "count"})
# Select 99% of components
component_sizes_pre = component_sizes_pre.sort_values("src", ascending=True).query(
    "src > 9"
)
component_sizes_pre = component_sizes_pre.iloc[: int(len(component_sizes_pre) * 0.99)]
component_sizes_pre["src"].plot.hist(bins=20, label="Normal")

# %%
graph_post = nx.from_pandas_edgelist(edges_post_df, "src", "dst")

# %%
# # Get connected components
# components_post = list(nx.connected_components(graph_post))
# # Assign components to edges
# edges_post_df["component"] = edges_post_df["src"].apply(
#     lambda x: [i for i, c in enumerate(components_post) if x in c][0]
# )

# %%
# Get component sizes
component_sizes_post = edges_post_df.groupby("component").agg(
    {"is_anomalous": "sum", "src": "count"}
)
# Select 99% of components
component_sizes_post = component_sizes_post.sort_values("src", ascending=True)
component_sizes_post = component_sizes_post.iloc[
    : int(len(component_sizes_post) * 0.99)
]
component_sizes_post["src"].plot.hist(bins=20, label="Normal")

# %%

# %%

# %%
# Get component sizes
component_sizes_post = edges_post_df.groupby(["component"]).agg(
    {"is_anomalous": "sum", "src": "count"}
)
component_sizes_post["has_anomaly"] = component_sizes_post["is_anomalous"] > 0
# Select 99% of components
component_sizes_post = component_sizes_post.sort_values("src", ascending=True)
component_sizes_post = component_sizes_post.iloc[
    : int(len(component_sizes_post) * 0.99)
]
anomaly_dist = component_sizes_post.groupby(["src", "has_anomaly"]).count().unstack()
# normalise
anomaly_dist = anomaly_dist / anomaly_dist.sum()
anomaly_dist.plot.hist(bins=20, alpha=0.5, label="Normal")

# %%
component_sizes_post

# %%
component_sizes_pre.groupby("src").size().sort_index()[:20]

# %%
component_sizes_post.groupby("src").size().sort_index()[:20]

# %%
component_sizes_post.groupby("src").agg(["sum", "count"])

# %% [markdown]
# ## Anomaly Simulation

# %%
graph_edges_post = edges_post_df.query("component == 23")
graph_edges_post

# %%
edges_pre_df.query("src == '2258222399048453312'")

# %%
graph_edges_pre = edges_pre_df.query("component == 77309426934").sort_values("src")

# %%
edges_in_pre_not_in_post = graph_edges_pre.query(
    "src not in @graph_edges_post.src.values"
)
edges_in_pre_not_in_post

# %%
graph_pre = nx.DiGraph()
graph_pre.add_nodes_from(graph_edges_pre["src"].unique())
graph_pre.add_nodes_from(graph_edges_pre["dst"].unique())
graph_pre.add_edges_from(graph_edges_pre[["src", "dst"]].values)

# %%
graph_post = nx.DiGraph()
graph_post.add_nodes_from(graph_edges_post["src"].unique())
graph_post.add_nodes_from(graph_edges_post["dst"].unique())
graph_post.add_edges_from(graph_edges_post[["src", "dst"]].values)

# %%
node_c = "lightblue"
new_node_c = "lightgreen"
dropped_node_c = "lightpink"
anomalous_node_c = "darkorange"

edge_c = "lightgrey"
new_edge_c = "green"
dropped_edge_c = "red"

# new_edge_c = edge_c
# dropped_edge_c = edge_c


# %%
graph_pre.nodes

# %%
i = 168
print(i)
pos = nx.drawing.spring_layout(graph_pre, seed=i, iterations=100, k=0.8)

NODE_SIZE = 100

# %%
anomaly_id = "2258222399048453312"
pos[anomaly_id] = [-0.40, 0.21]


# %%
def plot_pre(ax):

    edge_colours = [edge_c for u, v in graph_pre.edges]
    node_colours = [node_c for n in graph_pre.nodes]

    # pos = nx.drawing.spring_layout(graph_intermediate, iterations=1000, seed=i)
    nx.draw(
        graph_pre,
        pos=pos,
        edge_color=edge_colours,
        node_color=node_colours,
        # labels={n: n[:3] for n in graph_pre.nodes},
        node_size=NODE_SIZE,
        # font_size=8,
        # font_color="black",
        ax=ax,
    )


# %%
def plot_mid(ax):
    # Colour edges that are in the pre graph but not in the post graph
    edge_colours = [
        dropped_edge_c if (u) == "2258222399048453312" else edge_c
        for u, v in graph_pre.edges
    ]
    node_colours = [
        node_c if n in graph_post.nodes else dropped_node_c for n in graph_pre.nodes
    ]

    # Set colour for node 225
    idx = next(i for i, j in enumerate(graph_pre.nodes) if j == "2258222399048453312")
    node_colours[idx] = anomalous_node_c

    # pos = nx.drawing.spring_layout(graph_intermediate, iterations=1000, seed=i)
    nx.draw(
        graph_pre,
        pos=pos,
        edge_color=edge_colours,
        node_color=node_colours,
        # labels={n: n[:3] for n in graph_pre.nodes},
        node_size=NODE_SIZE,
        # font_size=8,
        # font_color="black",
        ax=ax,
    )


# %%
def plot_post(ax):
    pos2 = {**pos, "11395388473527546194": pos["17957350210417364187"]}

    # Colour edges in the post graph that are not in the pre graph
    edge_colours = [
        new_edge_c if u == "2258222399048453312" else edge_c
        for u, v in graph_post.edges
    ]
    node_colours = [
        node_c if n in graph_pre.nodes else new_node_c for n in graph_post.nodes
    ]

    # Set colour for node 225
    idx = next(i for i, j in enumerate(graph_post.nodes) if j == "2258222399048453312")
    node_colours[idx] = anomalous_node_c

    # pos = nx.drawing.spring_layout(graph_intermediate, iterations=1000, seed=i)
    nx.draw(
        graph_post,
        pos=pos2,
        edge_color=edge_colours,
        node_color=node_colours,
        # labels={n: n[:3] for n in graph_post.nodes},
        node_size=NODE_SIZE,
        # font_size=8,
        # font_color="black",
        ax=ax,
    )


# %%
fig, axes = plt.subplots(3, 1, figsize=(4, 12))
axes[0].set_title("Pre-Shuffle")
axes[1].set_title("Anomaly Assignment")
axes[2].set_title("Post-Shuffle")

plot_pre(axes[0])
plot_mid(axes[1])
plot_post(axes[2])

fig.savefig("figures/anomaly-simulation.png", dpi=300, bbox_inches="tight")


# %%

# %%
