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

# %% [markdown]
# ### Neo4j

# %%
next(Path.cwd().glob("data/processed/companies.parquet/part-00000*"))

# %%
import textwrap as tw
from pyspark.sql import DataFrame


def convert_to_csv(df: DataFrame, name: str) -> None:
    csv_path = f"data/processed/{name}.csv"
    parquet_csv_path = f"{csv_path}.parquet"
    df.coalesce(1).write.csv(parquet_csv_path, header=True, mode="ignore")
    csv_txt_path = next(Path.cwd().glob(f"{parquet_csv_path}/part-00000*.csv"))
    Path(csv_txt_path).rename(csv_path)


def make_neo4j_statement(df: DataFrame, name: str, entity_type: str) -> None:
    acc = []

    for col in df.columns:
        acc.append(f"{col}: row.{col}")

    statement = f"""\
        :auto USING PERIODIC COMMIT LOAD CSV WITH HEADERS FROM "file:///{name}.csv" AS row
        CREATE (:{entity_type.capitalize()} """
    statement += "{" + ", ".join(acc) + "})"
    statement = tw.dedent(statement)
    print(statement)
    print()


# %%
convert_to_csv(companies_processed_df, "companies")
convert_to_csv(persons_processed_df, "persons")
convert_to_csv(relationships_processed_df, "relationships")

make_neo4j_statement(companies_processed_df, "companies", "company")
make_neo4j_statement(persons_processed_df, "persons", "person")

# %%
relationships_processed_df.columns

# %% [markdown]
# ```cypher
# :auto USING PERIODIC COMMIT LOAD CSV WITH HEADERS FROM "file:///relationships.csv" AS row
# MATCH
#   (a:Person),
#   (b:Company)
# WHERE a.statementID = row.interestedPartyStatementID AND b.statementID = row.subjectStatementID
# CREATE (a)-[r:Ownership {minimumShare: row.minimumShare}]->(b)
# ```

# %% [markdown]
# ```cypher
# :auto USING PERIODIC COMMIT LOAD CSV WITH HEADERS FROM "file:///relationships.csv" AS row
# MATCH
#   (a:Company),
#   (b:Company)
# WHERE a.statementID = row.interestedPartyStatementID AND b.statementID = row.subjectStatementID
# CREATE (a)-[r:Ownership {minimumShare: row.minimumShare}]->(b)
# ```

# %%
companies_processed_df.coalesce(1).write.csv(
    "data/processed/companies.csv", header=True
)
persons_processed_df.coalesce(1).write.csv("data/processed/persons.csv", header=True)
relationships_processed_df.coalesce(1).write.csv(
    "data/processed/relationships.csv", header=True
)

# %% [markdown]
# ## Graph

# %%
graph = gf.GraphFrame(nodes_df, edges_df)

# %%
component_sizes = (
    connected_components.groupBy("component").count().orderBy(F.desc("count"))
)
component_sizes.count()

# %%
component_sizes.show(10)

# %%
large_components = component_sizes.filter("count >= 10 AND count <= 100")
large_components.count()

# %%
large_components.approxQuantile("count", [x for x in np.linspace(0, 1, 20)], 0.05)

# %%
large_df = large_components.toPandas()

# %%
_ = large_df["count"].plot.hist()

# %%
mega_component = connected_components.filter("component == 1863")

# %%
mega_component_nodes = nodes_df.join(mega_component, ["id", "isCompany"]).join(
    companies_processed_df, F.col("id") == F.col("statementID")
)
mega_component_nodes.select("name", "SICCode_SicText_1").show(truncate=False)

# %%
edges_df.show()

# %%
mega_graph = graph.filterVertices(
    F.col("id").isin([x.id for x in mega_component.select("id").collect()])
).cache()

# %%
mega_pr = mega_graph.pageRank(maxIter=20)

# %%
mega_pr.vertices.orderBy(F.desc("pagerank")).show()

# %%
query = "7094770822984538793"
companies_processed_df.filter(f"statementID == {query}").show(truncate=False)
persons_processed_df.filter(f"statementID == {query}").show(truncate=False)
relationships_processed_df.filter(
    f"interestedPartyStatementID == {query} OR subjectStatementID == {query}"
).show()

# %%
queries = [
    x.id for x in mega_pr.vertices.filter("pagerank >= 4").select("id").collect()
]
for query in queries:
    print(f"Query: {query}")
    print("-" * 79)
    companies_processed_df.filter(f"statementID == {query}").show(truncate=False)
    persons_processed_df.filter(f"statementID == {query}").show(truncate=False)
    relationships_processed_df.filter(
        f"interestedPartyStatementID == {query} OR subjectStatementID == {query}"
    ).show()
    print()

# %%
connected_components

# %%
connected_components.show()

# %%
large_component_ids = [
    row.component for row in large_components.select("component").collect()
]
component_nodes_df = connected_components.filter(
    F.col("component").isin(large_component_ids)
)

# %%
component_node_ids = [row.id for row in component_nodes_df.select("id").collect()]

# %%
filter_expr = (F.col("src").isin(component_node_ids)) | (
    F.col("dst").isin(component_node_ids)
)
component_edges_df = edges_df.filter(filter_expr)

# %%
join_expr = (
    component_nodes_df.id
    == component_edges_df.src | component_nodes_df.id
    == component_edges_df.dst
)
component_edges_df = component_edges_df.join(
    component_nodes_df, on=join_expr, how="inner"
)

# %%
large_component_node_df = nodes_df.filter(F.col("id").isin(large_component_id_list))
large_component_node_df.show()

# %%
large_component_edge_df = nodes_df.filter(F.col("id").isin(large_component_id_list))
large_component_edge_df.show()

# %% [markdown]
# ## Graph (OLD)

# %%
import igraph as ig

# %%
edges = relationships_processed_df.select(
    "interestedPartyStatementID", "subjectStatementID"
)
edges.count()

# %%
edge_list = [tuple(v for v in e) for e in edges.collect()]
# edge_list = [e for e in edges.collect()]

# %%
import itertools as it

graph = ig.Graph()
vertex_list = list(set(it.chain.from_iterable(edge_list)))
graph.add_vertices(vertex_list)
del vertex_list

# %%
graph.add_edges(edge_list)

# %%
clusters = graph.clusters("weak")
cluster_sizes = clusters.sizes()

# %%
subg_id = next(
    (idx, size) for idx, size in enumerate(cluster_sizes) if 200 < size < 500
)
subg = clusters.subgraph(subg_id[0])
subg.vcount()
ig.plot(subg)
