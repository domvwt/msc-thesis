"""
- Assign nodes to connected components
"""

from pathlib import Path

import yaml
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

from mscproject import dataprep as dp
from mscproject import graphprep as gp


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
    sc.setLogLevel("ERROR")
    spark = SparkSession.builder.config("spark.driver.memory", "8g").getOrCreate()

    # Load processed data.
    companies_processed_df = spark.read.parquet(conf_dict["companies_processed"])
    relationships_processed_df = spark.read.parquet(
        conf_dict["relationships_processed"]
    )
    persons_processed_df = spark.read.parquet(conf_dict["persons_processed"])

    # Create graph and get connected components.
    graph = gp.make_graph_from_entities_and_relationships(
        companies_processed_df, persons_processed_df, relationships_processed_df
    )
    connected_components = (
        graph.connectedComponents().select("id", "component", "isCompany").cache()
    )

    # Write outputs.
    output_path_map = {
        connected_components: conf_dict["connected_components"],
    }
    dp.write_if_missing_spark(output_path_map)

    # Stop session.
    spark.stop()


if __name__ == "__main__":
    main()