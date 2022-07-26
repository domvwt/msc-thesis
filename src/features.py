from platform import node

import networkx as nx
import pandas as pd


def make_graph(edges: pd.DataFrame) -> nx.DiGraph:
    edge_gen = ((row.src, row.dst) for row in edges.itertuples())
    graph = nx.DiGraph()
    graph.add_edges_from(edge_gen)
    return graph


# Get structural properties of nodes in the graph.
def get_node_features(graph: nx.DiGraph) -> pd.DataFrame:
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


# Get average neigbourhood properties of nodes in the graph.
def get_local_neighbourhood_features(
    graph: nx.DiGraph, node_features: pd.DataFrame, radius: int = 1
) -> pd.DataFrame:
    neighbourhood_feature_dict = {}

    def node_neighbour_features(node_id: int):
        if radius == 1:
            node_neighbours = list(graph.neighbors(node_id))
        else:
            node_neighbours = nx.generators.ego_graph(
                graph, node_id, radius, undirected=True
            ).nodes
        node_neighbour_features = node_features.loc[node_neighbours]
        sum_features = node_neighbour_features.sum()
        sum_features["num_neighbours"] = len(node_neighbours)
        return sum_features

    neighbourhood_feature_dict = {
        node: node_neighbour_features(node) for node in graph.nodes()
    }

    return (
        pd.DataFrame(neighbourhood_feature_dict)
        .T.rename(columns=lambda x: f"neighbourhood_{x}")
        .fillna(0)
    )
