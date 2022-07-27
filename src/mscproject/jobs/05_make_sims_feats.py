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

    # Sample 1000 records from each file if testing.
    if conf_dict["test_mode"]:
        print("Sampling 1000 records...")
        companies_nodes_df = companies_nodes_df.sample(1000)
        persons_nodes_df = persons_nodes_df.sample(1000)
        edges_df = edges_df.sample(1000)

    # Simulate anomalous nodes and edges.
    print("Simulating anomalous nodes and edges...")
    companies_anomalised, persons_anomalised, edges_anomalised = sim.simulate_anomalies(
        companies_nodes_df, persons_nodes_df, edges_df
    )

    # Generate graph features.
    print("Generating graph features...")
    graph_nx = feat.make_graph(edges_anomalised)
    node_features_df = feat.generate_node_features(graph_nx)
    neighbourhood_features_df = feat.get_local_neighbourhood_features(
        graph_nx, node_features_df
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

    # Save features.
    print("Saving features...")
    # ? Swapped dict for list of tuples as pd.DataFrame is unhashable.
    output_path_map = [
        (companies_nodes_with_features_df02, conf_dict["companies_features"]),
        (persons_nodes_with_features_df02, conf_dict["persons_features"]),
        (edges_anomalised, conf_dict["edges_features"]),
    ]
    dp.write_if_missing_pandas(output_path_map)
    print("Done.")


if __name__ == "__main__":
    main()
