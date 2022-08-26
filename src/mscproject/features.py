import functools as ft

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
    graph: nx.DiGraph, node_features: pd.DataFrame, radius: int, cc_map: dict
) -> pd.DataFrame:

    id_to_index_map = {node: i for i, node in enumerate(graph.nodes())}
    node_features["neighbour_count"] = 1
    node_features_np = node_features.to_numpy(dtype=np.float32)

    def node_ids_to_indices(node_id_list: list) -> np.ndarray:
        return np.array(
            [id_to_index_map[node_id] for node_id in node_id_list], dtype=np.int32
        )

    def node_neighbour_features(node_id: int):
        if radius == 1:
            node_neighbours = node_ids_to_indices(graph.neighbors(node_id))
        else:
            subgraph = cc_map[node_id]
            node_neighbours = node_ids_to_indices(
                nx.generators.ego_graph(
                    subgraph, node_id, radius, undirected=True
                ).nodes
            )
        return node_features_np[node_neighbours].sum(axis=0)

    neighbourhood_feature_dict = {
        node_id: node_neighbour_features(node_id) for node_id in graph.nodes
    }

    columns = [f"neighbourhood_{col}" for col in node_features.columns]
    return (
        pd.DataFrame(neighbourhood_feature_dict)
        .T.set_axis(columns, axis=1, inplace=False)
        .fillna(0)
    )


def get_local_neighbourhood_features_parallel(
    graph: nx.DiGraph,
    node_features: pd.DataFrame,
    radius: int,
    cc_map: dict,
    n_jobs: int = 8,
) -> pd.DataFrame:

    id_to_index_map = {node: i for i, node in enumerate(graph.nodes())}
    node_features["neighbour_count"] = 1
    node_features_np = node_features.to_numpy(dtype=np.float32)

    def node_ids_to_indices(node_id_list: list) -> np.ndarray:
        return np.array(
            [id_to_index_map[node_id] for node_id in node_id_list], dtype=np.int32
        )

    def node_neighbour_features(node_id: int):
        if radius == 1:
            node_neighbours = node_ids_to_indices(graph.neighbors(node_id))
        else:
            subgraph = cc_map[node_id]
            node_neighbours = node_ids_to_indices(
                nx.generators.ego_graph(
                    subgraph, node_id, radius, undirected=True
                ).nodes
            )
        return node_features_np[node_neighbours].sum(axis=0)

    # Get node neighbourhood features for batch of nodes.
    def node_neighbour_features_batch(node_id_list: list) -> dict:
        return {node_id: node_neighbour_features(node_id) for node_id in node_id_list}

    # Split nodes into evenly sized groups.
    node_groups = [
        list(graph.nodes())[i : i + len(graph.nodes()) // n_jobs]
        for i in range(0, len(graph.nodes()), len(graph.nodes()) // n_jobs)
    ]

    # Compute features for each group in parallel.
    neighbourhood_feature_dicts = [
        jl.Parallel(n_jobs=n_jobs)(
            jl.delayed(node_neighbour_features_batch)(node_group)
            for node_group in tqdm(node_groups)
        )
    ]

    # Concatenate all the feature dictionaries.
    neighbourhood_feature_dict = {}
    for d in neighbourhood_feature_dicts:
        for e in d:
            neighbourhood_feature_dict.update(e)

    columns = [f"neighbourhood_{col}" for col in node_features.columns]
    return (
        pd.DataFrame(neighbourhood_feature_dict)
        .T.set_axis(columns, axis=1, inplace=False)
        .fillna(0)
    )
