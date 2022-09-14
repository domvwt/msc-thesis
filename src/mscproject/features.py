import functools as ft
from typing import Optional

import joblib as jl
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


def make_graph(edges: pd.DataFrame) -> nx.DiGraph:
    edge_gen = ((row.src, row.dst) for row in edges.itertuples())
    graph = nx.DiGraph()
    graph.add_edges_from(edge_gen)
    return graph


# Get structural properties of nodes in the graph.
def generate_node_features(graph: nx.DiGraph) -> pd.DataFrame:
    feature_dict = dict()
    feature_dict["indegree"] = graph.in_degree()
    feature_dict["outdegree"] = graph.out_degree()
    feature_dict["closeness"] = nx.closeness_centrality(graph)
    # feature_dict["betweenness"] = nx.betweenness_centrality(graph)
    # nodes["eigenvector"] = nx.eigenvector_centrality(graph, tol=0.5)
    feature_dict["clustering"] = nx.clustering(graph)
    feature_dict["pagerank"] = nx.pagerank(graph)

    series_list = [
        pd.Series(dict(feature_dict[key]), name=key) for key in feature_dict.keys()
    ]
    return pd.concat(series_list, axis=1)


def get_node_to_cc_graph_map(graph: nx.DiGraph) -> dict:
    @ft.lru_cache(maxsize=None)
    def get_subgraph(cc_nodes):
        return graph.subgraph(cc_nodes)

    return {
        node_id: get_subgraph(frozenset(cc_nodes))
        for cc_nodes in nx.weakly_connected_components(graph)
        for node_id in cc_nodes
    }


def get_node_id_to_cc_id_map(graph: nx.DiGraph) -> dict:
    return {
        node_id: cc_id
        for cc_id, cc_nodes in enumerate(nx.weakly_connected_components(graph))
        for node_id in cc_nodes
    }


def get_local_neighbourhood_features(
    graph: nx.DiGraph,
    node_features: pd.DataFrame,
    radius: int,
    n_jobs: int = 8,
) -> pd.DataFrame:

    id_to_index_map = {node: i for i, node in enumerate(graph.nodes())}
    # node_features["neighbour_count"] = 1
    node_features_np = node_features.to_numpy(dtype=np.float32)

    def node_ids_to_indices(node_id_list: list) -> np.ndarray:
        return np.array(
            [id_to_index_map[node_id] for node_id in node_id_list], dtype=np.int32
        )

    def node_neighbour_features(node_id: int, cc_map: Optional[dict]) -> np.ndarray:
        if radius == 1:
            node_neighbours = node_ids_to_indices(graph.neighbors(node_id))
        else:
            assert cc_map is not None, "cc_map must be provided for radius > 1"
            subgraph = cc_map[node_id]
            node_neighbours = node_ids_to_indices(
                nx.generators.ego_graph(
                    subgraph, node_id, radius, undirected=True
                ).nodes
            )

        if node_neighbours.size == 0:
            node_neighbours = np.array([id_to_index_map[node_id]], dtype=np.int32)

        count_features = np.array([node_features_np[node_neighbours].shape[0]])
        min_features = node_features_np[node_neighbours].min(axis=0)
        max_features = node_features_np[node_neighbours].max(axis=0)
        sum_features = node_features_np[node_neighbours].sum(axis=0)
        mean_features = node_features_np[node_neighbours].mean(axis=0)
        std_features = node_features_np[node_neighbours].std(axis=0)

        result = np.concatenate(
            (
                count_features,
                min_features,
                max_features,
                sum_features,
                mean_features,
                std_features,
            )
        )

        if node_neighbours.size == 0:
            result[:] = 0

        return result

    if radius > 1:
        cc_map = get_node_to_cc_graph_map(graph)
    else:
        cc_map = None

    # Get node neighbourhood features for batch of nodes.
    def node_neighbour_features_batch(node_id_list: list) -> dict:
        return {
            node_id: node_neighbour_features(node_id, cc_map)
            for node_id in tqdm(node_id_list)
        }

    if n_jobs > 1:
        # Split nodes into evenly sized groups.
        node_groups = [
            list(graph.nodes())[i : i + len(graph.nodes()) // n_jobs]
            for i in range(0, len(graph.nodes()), len(graph.nodes()) // n_jobs)
        ]

        # Compute features for each group in parallel.
        neighbourhood_feature_dicts = [
            jl.Parallel(n_jobs=n_jobs)(
                jl.delayed(node_neighbour_features_batch)(node_group)
                for node_group in node_groups
            )
        ]

        # Concatenate all the feature dictionaries.
        consolidated_features_dict = {}
        for d in neighbourhood_feature_dicts:
            for e in d:
                consolidated_features_dict.update(e)
    else:
        consolidated_features_dict = node_neighbour_features_batch(list(graph.nodes()))

    columns = [
        f"neighbourhood_{col}_{agg}"
        for agg in ["min", "max", "sum", "mean", "std"]
        for col in node_features.columns
    ]

    columns = [
        "neighbourhood_count",
        *columns,
    ]

    return (
        pd.DataFrame(consolidated_features_dict)
        .T.set_axis(columns, axis=1, inplace=False)
        .fillna(0)
    )
