import pandas as pd
import networkx as nx


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
def get_average_neighbourhood_features(
    graph: nx.DiGraph, node_features: pd.DataFrame
) -> pd.DataFrame:
    neighbourhood_feature_dict = {}

    def node_neighbour_features(node_id: int):
        node_neighbours = graph.neighbors(node_id)
        node_neighbour_features = node_features.loc[node_neighbours]
        return node_neighbour_features.mean()

    neighbourhood_feature_dict = {
        node: node_neighbour_features(node) for node in graph.nodes()
    }

    return (
        pd.DataFrame(neighbourhood_feature_dict)
        .T.rename(columns=lambda x: f"neighbourhood_{x}")
        .fillna(0)
    )
