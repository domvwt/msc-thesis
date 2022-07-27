"""
- Resolve duplicated entities
- Filter for connected components with desired size and ratio of person nodes (0.1 = 1:10).
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
    companies_processed_df = spark.read.parquet(conf_dict["companies_interim_02"])
    relationships_processed_df = spark.read.parquet(
        conf_dict["relationships_interim_02"]
    )
    persons_processed_df = spark.read.parquet(conf_dict["persons_interim_02"])

    # Resolve duplicate nodes.
    connected_components_df = spark.read.parquet(conf_dict["connected_components"])
    nodes_df = gp.make_node_table(
        companies_processed_df, persons_processed_df, connected_components_df
    )
    edges_df = gp.make_edge_table(relationships_processed_df, connected_components_df)
    id_mapping_df = gp.map_duplicate_nodes_to_min_id(nodes_df)
    resolved_nodes_df, resolved_edges_df = gp.resolve_duplicate_nodes(
        nodes_df, edges_df, id_mapping_df
    )

    # Create graph.
    graph = gp.make_graph_from_nodes_and_edges(resolved_nodes_df, resolved_edges_df)

    # Filter for connected components with desired size and ratio of person nodes (0.1 = 1:10).
    graph_filtered = gp.filter_graph(
        graph, connected_components_df, 10, 1000, 0.1
    ).cache()
    nodes_filtered_df = graph_filtered.vertices.cache()
    edges_filtered_df = graph_filtered.edges

    # Separate companies nodes and persons nodes.
    companies_nodes_df = gp.get_companies_nodes(
        nodes_filtered_df, companies_processed_df
    )
    persons_nodes_df = gp.get_persons_nodes(nodes_filtered_df, persons_processed_df)

    # Write outputs.
    output_path_map = {
        companies_nodes_df: conf_dict["companies_nodes"],
        persons_nodes_df: conf_dict["persons_nodes"],
        edges_filtered_df: conf_dict["edges"],
    }
    dp.write_if_missing_spark(output_path_map)

    # Stop session.
    spark.stop()


if __name__ == "__main__":
    main()
