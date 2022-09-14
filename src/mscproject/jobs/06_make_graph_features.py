"""
- Simulate anomalous nodes / edges
- Create graph features
"""

from pathlib import Path

import pandas as pd
import yaml

import mscproject.dataprep as dp
import mscproject.features as feat


def main() -> None:
    # Read config.
    conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

    # Load graph data.
    print("Loading anomalised data...")
    companies_df = pd.read_parquet(conf_dict["companies_anomalies"])
    persons_df = pd.read_parquet(conf_dict["persons_anomalies"])
    edges_df = pd.read_parquet(conf_dict["edges_anomalies"])

    # Sample records if sample size is specified.
    if conf_dict["sample_size"] is not None:
        print(f"Sampling {conf_dict['sample_size']} records...")
        companies_df = companies_df.sample(conf_dict["sample_size"])
        persons_df = persons_df.sample(conf_dict["sample_size"])
        edges_df = edges_df.sample(conf_dict["sample_size"])

    # Generate graph features.
    print("Generating graph features...")
    graph_nx = feat.make_graph(edges_df)
    node_features_df = feat.generate_node_features(graph_nx)
    print("Generating local neighbourhood features...")
    neighbourhood_features_df = feat.get_local_neighbourhood_features(
        graph=graph_nx, node_features=node_features_df, radius=1, n_jobs=1
    )

    # Make index into id column.
    node_features_df["id"] = node_features_df.index
    neighbourhood_features_df["id"] = neighbourhood_features_df.index

    # Join features to nodes.
    print("Joining features to nodes...")
    companies_nodes_with_features_df01 = pd.merge(
        companies_df, node_features_df, how="left", on="id"
    )
    companies_nodes_with_features_df02 = pd.merge(
        companies_nodes_with_features_df01,
        neighbourhood_features_df,
        how="left",
        on="id",
    )
    persons_nodes_with_features_df01 = pd.merge(
        persons_df, node_features_df, how="left", on="id"
    )
    persons_nodes_with_features_df02 = pd.merge(
        persons_nodes_with_features_df01,
        neighbourhood_features_df,
        how="left",
        on="id",
    )

    # Save features.
    print("Saving features...")
    output_path_map = [
        (companies_nodes_with_features_df02, conf_dict["companies_features"]),
        (persons_nodes_with_features_df02, conf_dict["persons_features"]),
        (edges_df, conf_dict["edges_features"]),
    ]
    dp.write_output_path_map(output_path_map)
    print("Done.")


if __name__ == "__main__":
    main()
