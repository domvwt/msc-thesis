"""
- Simulate anomalous nodes / edges
- Create graph features
"""

from pathlib import Path

import pandas as pd
import yaml

import mscproject.dataprep as dp
import mscproject.features as feat
import mscproject.simulate as sim


def main() -> None:
    # Read config.
    conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

    # Load graph data.
    print("Loading graph data...")
    companies_nodes_df = pd.read_parquet(conf_dict["companies_nodes"])
    persons_nodes_df = pd.read_parquet(conf_dict["persons_nodes"])
    edges_df = pd.read_parquet(conf_dict["edges"])

    # Sample records if sample size is specified.
    if conf_dict["sample_size"] is not None:
        print(f"Sampling {conf_dict['sample_size']} records...")
        companies_nodes_df = companies_nodes_df.sample(conf_dict["sample_size"])
        persons_nodes_df = persons_nodes_df.sample(conf_dict["sample_size"])
        edges_df = edges_df.sample(conf_dict["sample_size"])

    # Simulate anomalous nodes and edges.
    print("Simulating anomalous nodes and edges...")
    companies_anomalised, persons_anomalised, edges_anomalised = sim.simulate_anomalies(
        companies_nodes_df, persons_nodes_df, edges_df
    )

    # Generate graph features.
    print("Generating graph features...")
    graph_nx = feat.make_graph(edges_anomalised)
    node_features_df = feat.generate_node_features(graph_nx)
    # neighbourhood_features_df = feat.get_local_neighbourhood_features_fast(
    #     graph_nx, node_features_df, 2
    # )
    print("Generating connected component map...")
    cc_map = feat.get_node_to_cc_graph_map(graph_nx)
    print("Generating local neighbourhood features...")
    neighbourhood_features_df = feat.get_local_neighbourhood_features_parallel(
        graph_nx, node_features_df, 2, cc_map
    )

    # Make index into id column.
    node_features_df["id"] = node_features_df.index
    neighbourhood_features_df["id"] = neighbourhood_features_df.index

    # Join features to nodes.
    print("Joining features to nodes...")
    companies_nodes_with_features_df01 = pd.merge(
        companies_anomalised, node_features_df, how="left", on="id"
    )
    companies_nodes_with_features_df02 = pd.merge(
        companies_nodes_with_features_df01,
        neighbourhood_features_df,
        how="left",
        on="id",
    )
    persons_nodes_with_features_df01 = pd.merge(
        persons_anomalised, node_features_df, how="left", on="id"
    )
    persons_nodes_with_features_df02 = pd.merge(
        persons_nodes_with_features_df01,
        neighbourhood_features_df,
        how="left",
        on="id",
    )

    # Get new component ids.
    print("Updating component ids...")
    new_graph = feat.make_graph(edges_anomalised)
    node_id_to_cc_id_map = feat.get_node_id_to_cc_id_map(new_graph)
    companies_nodes_with_features_df02[
        "component"
    ] = companies_nodes_with_features_df02["id"].map(node_id_to_cc_id_map)
    persons_nodes_with_features_df02["component"] = persons_nodes_with_features_df02[
        "id"
    ].map(node_id_to_cc_id_map)
    edges_anomalised["component"] = edges_anomalised["src"].map(node_id_to_cc_id_map)

    # Save features.
    print("Saving features...")
    # ? Swapped dict for list of tuples as pd.DataFrame is unhashable.
    output_path_map = [
        (companies_nodes_with_features_df02, conf_dict["companies_features"]),
        (persons_nodes_with_features_df02, conf_dict["persons_features"]),
        (edges_anomalised, conf_dict["edges_features"]),
    ]
    dp.write_output_path_map(output_path_map)
    print("Done.")


if __name__ == "__main__":
    main()
