from typing import Tuple

import graphframes as gf
import pyspark.sql.functions as F
from graphframes import GraphFrame
from pyspark.sql import DataFrame


def make_graph_from_entities_and_relationships(
    companies_processed_df: DataFrame,
    persons_processed_df: DataFrame,
    relationships_processed_df: DataFrame,
) -> GraphFrame:
    """Create a graph from the entities and relationships dataframes."""
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
    return gf.GraphFrame(nodes_df, edges_df)


def make_node_table(
    companies_processed_df: DataFrame,
    persons_processed_df: DataFrame,
    connected_components_df: DataFrame,
) -> DataFrame:
    """Create a node table from the entities and connected components dataframes."""

    def helper(entities_processed_df: DataFrame, is_company: bool) -> DataFrame:
        return entities_processed_df.withColumnRenamed("statementID", "id").select(
            "id", "name", F.lit(is_company).alias("isCompany")
        )

    companies = helper(companies_processed_df, True)
    persons = helper(persons_processed_df, False)
    nodes_df = companies.union(persons)
    return nodes_df.join(
        connected_components_df.select("id", "component"), ["id"], "inner"
    )


def make_edge_table(
    relationships_processed_df: DataFrame, connected_components_df: DataFrame
) -> DataFrame:
    """Create an edge table from the relationships and connected components dataframes."""
    return (
        relationships_processed_df.withColumnRenamed(
            "interestedPartyStatementID", "src"
        )
        .withColumnRenamed("subjectStatementID", "dst")
        .join(
            connected_components_df.select(F.col("id").alias("src"), "component"),
            ["src"],
            "inner",
        )
    )


def make_graph_from_nodes_and_edges(
    node_table: DataFrame, edge_table: DataFrame
) -> GraphFrame:
    """Create a graph from the node and edge tables."""
    return GraphFrame(node_table, edge_table)


def get_components_with_size(
    connected_components_df: DataFrame,
    min_component_nodes: int,
    max_component_nodes: int,
) -> DataFrame:
    """Get the components with a size between the given bounds."""
    return (
        (connected_components_df.groupBy("component").count())
        .filter(
            (F.col("count") >= min_component_nodes)
            & (F.col("count") <= max_component_nodes)
        )
        .select("component")
    )


def get_components_with_persons(
    connected_components_df: DataFrame, person_ratio: float
) -> DataFrame:
    """Get the components with a ratio of persons to companies above the given threshold."""
    return (
        connected_components_df.groupBy("component", "isCompany")
        .count()
        .groupBy("component")
        .pivot("isCompany", ["true", "false"])
        .agg(F.sum("count"))
        .withColumn("person_ratio", F.col("false") / F.col("true"))
        .filter(F.col("person_ratio") >= person_ratio)
        .select("component")
    )


def filter_graph(
    graph: GraphFrame,
    connected_components_df: DataFrame,
    min_component_nodes: int,
    max_component_nodes: int,
    min_ratio_persons: float,
) -> GraphFrame:
    """Filter the graph to only include nodes and edges in the given components."""
    large_components = get_components_with_size(
        connected_components_df, min_component_nodes, max_component_nodes
    )
    persons_components = get_components_with_persons(
        connected_components_df, min_ratio_persons
    )
    target_component_ids = large_components.join(
        persons_components, ["component"], "inner"
    )
    nodes = graph.vertices.join(target_component_ids, ["component"], "inner")
    edges = graph.edges.join(target_component_ids, ["component"], "inner")
    return GraphFrame(nodes, edges).dropIsolatedVertices()


def get_companies_nodes(
    nodes_df: DataFrame, companies_processed_df: DataFrame
) -> DataFrame:
    """Get the companies nodes from the nodes dataframe."""
    return nodes_df.filter(F.col("isCompany")).join(
        companies_processed_df.withColumnRenamed("statementID", "id"), ["id"], "inner"
    )


def get_persons_nodes(
    nodes_df: DataFrame, persons_processed_df: DataFrame
) -> DataFrame:
    """Get the persons nodes from the nodes dataframe."""
    return nodes_df.filter(~F.col("isCompany")).join(
        persons_processed_df.withColumnRenamed("statementID", "id"), ["id"], "inner"
    )


def map_duplicate_nodes_to_min_id(
    nodes_df: DataFrame,
) -> DataFrame:
    """Map duplicate nodes to the minimum id."""
    return (
        nodes_df.groupBy("name", "component")
        .agg(F.count("*").alias("count"), F.min("id").alias("id_new"))
        .filter(F.col("count") > 1)
        .join(nodes_df, ["name", "component"], "inner")
        .select(F.col("id"), F.col("id_new"))
    )


def resolve_duplicate_nodes(
    nodes_df: DataFrame,
    edges_df: DataFrame,
    id_mapping_df: DataFrame,
) -> Tuple[DataFrame, DataFrame]:
    """Resolve duplicate nodes by mapping them to the minimum id and updating the edges."""
    node_columns = ["id", "isCompany", "component"]
    resolved_nodes_df = (
        nodes_df.join(id_mapping_df, ["id"], "left")
        .withColumn("id", F.coalesce(F.col("id_new"), F.col("id")))
        .select(node_columns)
        .dropDuplicates()
    )
    edge_columns = [
        "src",
        "dst",
        "interestedPartyIsPerson",
        "minimumShare",
        "component",
    ]
    resolved_edges_df = (
        edges_df.join(id_mapping_df.withColumnRenamed("id", "src"), ["src"], "left")
        .withColumn("src", F.coalesce(F.col("id_new"), F.col("src")))
        .drop("id_new")
        .join(id_mapping_df.withColumnRenamed("id", "dst"), ["dst"], "left")
        .withColumn("dst", F.coalesce(F.col("id_new"), F.col("dst")))
        .groupBy("src", "dst")
        .agg(
            F.first("interestedPartyIsPerson").alias("interestedPartyIsPerson"),
            F.max("minimumShare").alias("minimumShare"),
            F.first("component").alias("component"),
        )
        .select(edge_columns)
    )
    return resolved_nodes_df, resolved_edges_df
