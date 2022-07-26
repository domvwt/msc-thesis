from pathlib import Path

import pyspark.sql.functions as F
import yaml
from pyspark import SparkConf, SparkContext
from pyspark.sql import Column, DataFrame, SparkSession

import mscproject.dataprep as dp


def main() -> None:
    conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

    spark = SparkSession.builder.config("spark.driver.memory", "8g").getOrCreate()

    companies_nodes_df = spark.read.parquet(conf_dict["companies_nodes"])
    persons_nodes_df = spark.read.parquet(conf_dict["persons_nodes"])
    edges_df = spark.read.parquet(conf_dict["edges"])

    select_cols = ["id", "name", "component", "isCompany"]
    nodes_union_df = companies_nodes_df.select(select_cols).union(
        persons_nodes_df.select(select_cols)
    )

    gephi_nodes_df = nodes_union_df.withColumnRenamed("name", "label")
    gephi_edges_df = edges_df.withColumnRenamed("src", "source").withColumnRenamed(
        "dst", "target"
    )

    # Write gephi dataframes to csv.
    dp.write_csv(gephi_nodes_df, "data/gephi", "nodes.csv")
    dp.write_csv(gephi_edges_df, "data/gephi", "edges.csv")


if __name__ == "__main__":
    main()
