import textwrap
from pathlib import Path

import pyspark.sql.functions as F
import yaml
from pyspark import SparkConf, SparkContext
from pyspark.sql import Column, DataFrame, SparkSession


def main() -> None:
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

    companies_processed_df = spark.read.parquet(conf_dict["companies_interim_02"])
    persons_processed_df = spark.read.parquet(conf_dict["persons_interim_02"])
    nodes_filtered_df = spark.read.parquet(conf_dict["nodes"])
    edges_filtered_df = spark.read.parquet(conf_dict["edges"])

    companies_filtered_df = join_graph_features_to_entities(
        companies_processed_df, nodes_filtered_df
    )
    persons_filtered_df = join_graph_features_to_entities(
        persons_processed_df, nodes_filtered_df
    )

    convert_to_csv(companies_filtered_df, "companies")
    convert_to_csv(persons_filtered_df, "persons")
    convert_to_csv(edges_filtered_df, "relationships")


def sanitise_string(column: str) -> Column:
    """Replace single quotes with double for neo4j import compatibility."""
    return F.regexp_replace(
        F.regexp_replace(F.col(column), pattern='"', replacement="'"),
        pattern="'",
        replacement="''",
    )


def join_graph_features_to_entities(
    entity_df: DataFrame, nodes_df: DataFrame
) -> DataFrame:
    nodes_df = nodes_df.drop("isCompany").withColumnRenamed("id", "statementID")
    return entity_df.join(nodes_df, on=["statementID"], how="inner")


def convert_to_csv(df: DataFrame, output_name: str) -> None:
    csv_path = f"data/neo4j/{output_name}.csv"
    parquet_csv_path = f"{csv_path}.parquet"
    string_cols_sanitised = [
        sanitise_string(col[0]).alias(col[0]) if col[1] == "string" else F.col(col[0])
        for col in df.dtypes
    ]
    sanitised_df = df.select(*string_cols_sanitised)
    sanitised_df.coalesce(1).write.csv(
        parquet_csv_path, header=False, escape="", mode="overwrite"
    )
    csv_txt_path = next(Path.cwd().glob(f"{parquet_csv_path}/part-00000*.csv"))
    Path(csv_txt_path).rename(csv_path)


def make_neo4j_statement(df: DataFrame, filename: str, entity_type: str) -> None:
    """Write cypher query to upload nodes to neo4j.

    Uploading data in this way is only suitable for datasets.
    Use the `neo4j-admin import` command for bulk loads.

    Args:
        df (DataFrame): Node DataFrame.
        filename (str): Also used as output file name.
        entity_type (str): Label to be applied to nodes.
    """
    acc = []

    for col in df.columns:
        acc.append(f"{col}: row.{col}")

    statement = f"""\
        :auto USING PERIODIC COMMIT LOAD CSV WITH HEADERS FROM "file:///{filename}.csv" AS row
        CREATE (:{entity_type.capitalize()} """
    statement += "{" + ", ".join(acc) + "});"
    statement = textwrap.dedent(statement)
    print(statement)
    print()


if __name__ == "__main__":
    main()
