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
from pyspark.sql import SparkSession, Column

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
nodes_filtered_df = spark.read.parquet("data/graph/component-nodes.parquet")
edges_filtered_df = spark.read.parquet("data/graph/component-edges.parquet")

# %%
print(f"Node count: {nodes_filtered_df.count():,}")
print(f"Edge count: {edges_filtered_df.count():,}")

# %%
nodes_filtered_df.groupBy("isCompany").count().show()

# %% [markdown]
# ### Neo4j

# %%
import textwrap as tw
from pyspark.sql import DataFrame


def sanitise_string(column: str) -> Column:
    """Replace single quotes with double for neo4j import compatibility."""
    return F.regexp_replace(
        F.regexp_replace(F.col(column), pattern='"', replacement="'"),
        pattern="'",
        replacement="''",
    )


def convert_to_csv(df: DataFrame, output_name: str) -> None:
    csv_path = f"data/neo4j/{output_name}.csv"
    parquet_csv_path = f"{csv_path}.parquet"
    string_cols_sanitised = [
        sanitise_string(col[0]).alias(col[0]) if col[1] == "string" else col[0]
        for col in df.dtypes
    ]
    sanitised_df = df.select(*string_cols_sanitised)
    sanitised_df.coalesce(1).write.csv(
        parquet_csv_path, header=False, escape="", mode="overwrite"
    )
    csv_txt_path = next(Path.cwd().glob(f"{parquet_csv_path}/part-00000*.csv"))
    Path(csv_txt_path).rename(csv_path)


def make_neo4j_statement(df: DataFrame, name: str, entity_type: str) -> None:
    acc = []

    for col in df.columns:
        acc.append(f"{col}: row.{col}")

    statement = f"""\
        :auto USING PERIODIC COMMIT LOAD CSV WITH HEADERS FROM "file:///{name}.csv" AS row
        CREATE (:{entity_type.capitalize()} """
    statement += "{" + ", ".join(acc) + "});"
    statement = tw.dedent(statement)
    print(statement)
    print()


# %%
def join_graph_features_to_entities(
    entity_df: DataFrame, nodes_df: DataFrame
) -> DataFrame:
    nodes_df = nodes_df.drop("isCompany").withColumnRenamed("id", "statementID")
    return entity_df.join(nodes_df, on=["statementID"], how="inner")


# %%
nodes_filtered_df.filter(~F.col("isCompany")).show()

# %%
companies_filtered_df = join_graph_features_to_entities(
    companies_processed_df, nodes_filtered_df
)
persons_filtered_df = join_graph_features_to_entities(
    persons_processed_df, nodes_filtered_df
)

# %% [markdown]
# # TODO
#
# - Connect company information back to nodes df to make companies nodes df
# - Connect person information back to nodes df to make persons nodes df
#

# %%
convert_to_csv(companies_filtered_df, "companies")
convert_to_csv(persons_filtered_df, "persons")
convert_to_csv(edges_filtered_df, "relationships")

# %%
make_neo4j_statement(companies_filtered_df, "companies", "company")
make_neo4j_statement(persons_filtered_df, "persons", "person")

# %%
persons_filtered_df.printSchema()

# %%
edges_filtered_df.printSchema()

# %% [markdown]
# ```cypher
# :auto USING PERIODIC COMMIT LOAD CSV WITH HEADERS FROM "file:///relationships.csv" AS row
# MATCH
#   (a:Person),
#   (b:Company)
# WHERE a.statementID = row.interestedPartyStatementID AND b.statementID = row.subjectStatementID
# CREATE (a)-[r:Ownership {minimumShare: row.minimumShare}]->(b);
#
# :auto USING PERIODIC COMMIT LOAD CSV WITH HEADERS FROM "file:///relationships.csv" AS row
# MATCH
#   (a:Company),
#   (b:Company)
# WHERE a.statementID = row.interestedPartyStatementID AND b.statementID = row.subjectStatementID
# CREATE (a)-[r:Ownership {minimumShare: row.minimumShare}]->(b)
# ```
