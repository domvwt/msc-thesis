import graphframes as gf
import pyspark.sql.functions as F
from graphframes import GraphFrame
from pyspark.sql import DataFrame


def make_graph(
    companies_processed_df: DataFrame,
    persons_processed_df: DataFrame,
    relationships_processed_df: DataFrame,
) -> GraphFrame:
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


def filter_graph(
    graph: GraphFrame, connected_components: DataFrame, min_component_nodes: int
) -> GraphFrame:
    large_components = (connected_components.groupBy("component").count()).filter(
        F.col("count") >= min_component_nodes
    )
    large_component_ids = [
        row.component for row in large_components.select("component").collect()
    ]
    return (
        graph.filterVertices(F.col("component").isin(large_component_ids))
        .dropIsolatedVertices()
        .cache()
    )


def derive_node_features(graph: GraphFrame) -> DataFrame:
    return (
        graph.vertices.join(graph.inDegrees, ["id"], how="left")
        .join(graph.outDegrees, ["id"], how="left")
        .join(
            graph.triangleCount()
            .withColumnRenamed("count", "triangleCount")
            .select("id", "triangleCount"),
            ["id"],
        )
        .fillna(0)
    )
