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

import graphframes as gf
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

while not Path("data") in Path(".").iterdir():
    os.chdir("..")

plt.style.use("seaborn-white")
conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

checkpoint_dir = str(Path("spark-checkpoints").absolute())
graphframes_jar_path = str(
    Path(
        ".venv/lib/python3.9/site-packages/pyspark/jars/graphframes-0.8.2-spark3.1-s_2.12.jar"
    ).absolute()
)

spark_conf = (
    SparkConf()
    .set("spark.jars", graphframes_jar_path)
    .set("spark.sql.sources.partitionOverwriteMode", "dynamic")
)

sc = SparkContext(conf=spark_conf).getOrCreate()
sc.setCheckpointDir(checkpoint_dir)
sc.setLogLevel("ERROR")

spark = SparkSession.builder.config("spark.driver.memory", "8g").getOrCreate()

# %%
companies_processed_df = spark.read.parquet(conf_dict["companies_interim_02"])
relationships_processed_df = spark.read.parquet(conf_dict["relationships_interim_02"])
persons_processed_df = spark.read.parquet(conf_dict["persons_interim_02"])
nodes_df = spark.read.parquet(conf_dict["nodes"])
edges_df = spark.read.parquet(conf_dict["edges"])
connected_components = spark.read.parquet(conf_dict["connected_components"])

# %%
print(f"Node count: {nodes_df.count():,}")
print(f"Edge count: {edges_df.count():,}")

# %% [markdown]
# ## Graph

# %%
graph = gf.GraphFrame(connected_components, edges_df)

# %%
component_sizes = (
    connected_components.groupBy("component").count().orderBy(F.desc("count"))
)
print(f"Connected component count: {component_sizes.count():,}")

# %%
component_sizes.filter("count <= 300").show()

# %%
large_components = component_sizes.filter("count >= 10")
large_components.count()
print(f"Large component count: {large_components.count():,}")

# %%
large_component_ids = [
    row.component for row in large_components.select("component").collect()
]
graph_filtered = (
    graph.filterVertices(F.col("component").isin(large_component_ids))
    .dropIsolatedVertices()
    .cache()
)

# %%
edges_filtered_df = graph_filtered.edges
edges_filtered_df.write.parquet("data/graph/component-edges.parquet", mode="overwrite")

# %%
graph_filtered.vertices.groupBy("isCompany").count().show()

# %%
# graph_pageranked = graph_filtered.pageRank(resetProbability=0.1, maxIter=20)
# nodes_pageranked = graph_pageranked.vertices.select("id", F.col("pagerank").cast("Long"))

# %%
nodes_filtered_df = (
    graph_filtered.vertices.join(graph_filtered.inDegrees, ["id"], how="left")
    .join(graph_filtered.outDegrees, ["id"], how="left")
    .join(
        graph_filtered.triangleCount()
        .withColumnRenamed("count", "triangleCount")
        .select("id", "triangleCount"),
        ["id"],
    )
    # .join(nodes_pageranked, ["id"])
    .fillna(0)
)
nodes_filtered_df.write.parquet("data/graph/component-nodes.parquet", mode="overwrite")
