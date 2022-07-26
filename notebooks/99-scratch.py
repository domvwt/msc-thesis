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
companies_nodes_df = spark.read.parquet(conf_dict["companies_nodes"])
persons_nodes_df = spark.read.parquet(conf_dict["persons_nodes"])
edges_df = spark.read.parquet(conf_dict["edges"])

# %%
print(f"Companies count: {companies_nodes_df.count():,}")
print(f"Persons count: {persons_nodes_df.count():,}")
print(f"Edge count: {edges_df.count():,}")

# %%
select_cols = ["id", "component", "isCompany"]
all_nodes = companies_nodes_df.select(select_cols).union(
    persons_nodes_df.select(select_cols)
)

# %%
all_nodes

# %%
component_summary = (
    all_nodes.groupBy("component", "isCompany")
    .count()
    .groupBy("component")
    .pivot("isCompany", ["true", "false"])
    .agg(F.sum("count"))
    .withColumnRenamed("true", "company_count")
    .withColumnRenamed("false", "person_count")
    .withColumn("person_proportion", F.col("person_count") / F.col("company_count"))
    .fillna(0)
    .orderBy("person_proportion", ascending=False)
)
component_summary.show()

# %%
component_summary.orderBy("person_proportion").show()

# %%
components_summary_pd = component_summary.toPandas()

# %%
components_summary_pd.query("person_proportion > 0.1 & person_proportion < 1")[
    "person_proportion"
].plot.hist(bins=10, alpha=0.5)

# %%
components_summary_pd

# %%
components_summary_pd["person_count"].plot.hist(bins=50, alpha=0.5)

# %%
components_summary_pd.query("person_count > 0").tail()

# %%
companies_nodes_df.filter(F.col("component") == "8589942400").toPandas()

# %%
persons_nodes_df.filter(F.col("component") == "8589942400").toPandas()


# %%
companies_nodes_df.filter(F.col("component").isNull()).toPandas()

# %%
companies_nodes_df.columns

# %%
companies_nodes_df.filter(F.col("name").isNull()).toPandas()

# %%

# %%

# %%
