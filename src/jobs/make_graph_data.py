from pathlib import Path

import yaml
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

from src import dataprep as dp
from src import graphprep as gp


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
    graph = gp.make_graph(
        companies_processed_df, persons_processed_df, relationships_processed_df
    )
    connected_components = graph.connectedComponents().select("id", "component").cache()

    # Filter for connected components where n >= 10 and n <= 50.
    graph_filtered = gp.filter_graph(graph, connected_components, 10, 50).cache()
    nodes_filtered_df = graph_filtered.vertices
    edges_filtered_df = graph_filtered.edges

    # Write outputs.
    output_path_map = {
        nodes_filtered_df: conf_dict["nodes"],
        edges_filtered_df: conf_dict["edges"],
        connected_components: conf_dict["connected_components"],
    }
    dp.write_if_missing(output_path_map)


if __name__ == "__main__":
    main()
