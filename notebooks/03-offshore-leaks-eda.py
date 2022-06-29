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

from pyspark import SparkContext
from pyspark.sql import SparkSession

while not Path("data") in Path(".").iterdir():
    os.chdir("..")

plt.style.use("seaborn-white")
conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

sc = SparkContext().getOrCreate()

spark = SparkSession.builder.config("spark.driver.memory", "8g").getOrCreate()

# %%
entities_df = pd.read_csv("data/raw/offshore-leaks/nodes-entities.csv")

# %%
entities_df.info()

# %%
entities_df.head().T

# %%
entities_df.countries.value_counts()

# %%
entities_df.jurisdiction_description.value_counts()[:10]

# %%
entities_df[entities_df.jurisdiction.str.contains("GBR").fillna(False)]

# %%
entities_df[entities_df.country_codes.str.contains("GBR").fillna(False)].iloc[:10, :]
