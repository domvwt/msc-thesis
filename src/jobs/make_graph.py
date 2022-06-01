from pathlib import Path

import graphframes as gf
import pyspark.sql.functions as F
import yaml
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

import src.dataprep as dp


def main() -> None:
    # Read config.
    conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

    # Start session.
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
    spark = SparkSession.builder.config("spark.driver.memory", "8g").getOrCreate()

    # Load processed data.
    companies_processed_df = spark.read.parquet(conf_dict["companies_processed"])
    relationships_processed_df = spark.read.parquet(
        conf_dict["relationships_processed"]
    )
    persons_processed_df = spark.read.parquet(conf_dict["persons_processed"])
    addresses_processed_df = spark.read.parquet(conf_dict["addresses_processed"])

    # Create graph.
    company_nodes_df = companies_processed_df.withColumnRenamed(
        "statementID", "id"
    ).select("id", F.lit(True).alias("isCompany"))
    person_nodes_df = persons_processed_df.withColumnRenamed(
        "statementID", "id"
    ).select("id", F.lit(False).alias("isCompany"))
    nodes_df = company_nodes_df.union(person_nodes_df)
    edges_df = relationships_processed_df.withColumnRenamed(
        "interestedPartyStatementID", "src"
    ).withColumnRenamed("subjectStatementID", "dst")
    graph = gf.GraphFrame(nodes_df, edges_df)
    connected_components = graph.connectedComponents()

    # Write outputs.
    output_path_map = {
        nodes_df: conf_dict["nodes"],
        edges_df: conf_dict["edges"],
        connected_components: conf_dict["connected_components"],
    }
    dp.write_if_missing(output_path_map)


if __name__ == "__main__":
    main()
