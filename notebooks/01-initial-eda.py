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

# %% [markdown]
# # Exploratory Data Analysis

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
import yaml

from pandas_profiling import ProfileReport
from pyspark.sql import SparkSession

while not Path("data") in Path(".").iterdir():
    os.chdir("..")

plt.style.use("seaborn-white")
conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

graphframes_jar_path = str(
    Path("jars/graphframes-0.8.2-spark3.1-s_2.12.jar").absolute()
)

spark = (
    SparkSession.builder.config("spark.driver.memory", "8g")
    .config(
        "spark.jars",
        graphframes_jar_path,
    )
    .getOrCreate()
)

# %%
companies_processed_df = spark.read.parquet(conf_dict["companies_interim_02"])
relationships_processed_df = spark.read.parquet(conf_dict["relationships_interim_02"])
persons_processed_df = spark.read.parquet(conf_dict["persons_interim_02"])
addresses_processed_df = spark.read.parquet(conf_dict["addresses_interim_02"])

# %%
all_dfs_dict = {
    "companies": companies_processed_df,
    "relationships": relationships_processed_df,
    "persons": persons_processed_df,
    "addresses": addresses_processed_df,
}

# %%
for key, df in all_dfs_dict.items():
    print(key)
    print("-" * 79)
    print(f"{df.count():,}")
    df.printSchema()
    print()

# %% [markdown]
# ## Profiles

# %%
all_pd_dict = {}

for name, df in all_dfs_dict.items():
    path = f"notebooks/{name}.html"
    df_pd: pd.DataFrame = df.sample(0.001).toPandas().sample(1_000)  # type: ignore
    all_pd_dict[name] = df_pd
    if not Path(path).exists():
        profile = ProfileReport(df_pd)
        profile.to_file(path)
    else:
        print(f"file exists: {path}; profiling skipped...")


# %% [markdown]
# ## Missing Values

# %%
def plot_missing(df):
    nulls = df.isna().to_numpy(dtype=int).T
    plt.imshow(nulls, aspect="auto", interpolation="none")
    plt.yticks(ticks=range(df.shape[1]), labels=df.columns)
    plt.show()


# %%
for name, df in all_pd_dict.items():
    print(name)
    print("-" * 79)
    plot_missing(df)
    print()
    print()

# %% [markdown]
# ## Companies
#
# Quite a few companies are missing information from Companies House. Further investigation through Companies House suggests these are largely dissolved or dormant companies.

# %%
grouping = (
    F.col("CompaniesHouseID").isNotNull().alias("hasChID"),
    F.col("SICCode_SicText_1").isNull().alias("hasSicCode"),
    F.col("CompanyStatus").contains("Active").alias("active"),
)
companies_processed_df.groupBy(*grouping).count().orderBy(*grouping).show()

# %%
activity_null = (
    companies_processed_df.filter(
        (F.col("SICCode_SicText_1").isNull())
        & (F.col("CompaniesHouseID").isNotNull())
        & (F.col("CompanyStatus").isNull())
    )
    .limit(10)
    .toPandas()
)
for row in list(activity_null.itertuples())[:5]:
    print(row)
    print()

# %%
no_ch_id = (
    companies_processed_df.filter((F.col("CompaniesHouseID").isNull()))
    .limit(10)
    .toPandas()
)
for row in list(no_ch_id.itertuples())[:5]:
    print(row)
    print()
