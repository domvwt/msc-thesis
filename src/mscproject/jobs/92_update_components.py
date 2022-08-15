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
    companies_nodes_with_features_df02 = pd.read_parquet(
        conf_dict["companies_features"]
    )
    persons_nodes_with_features_df02 = pd.read_parquet(conf_dict["persons_features"])
    edges_anomalised = pd.read_parquet(conf_dict["edges_features"])

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
    output_path_map = (
        (companies_nodes_with_features_df02, conf_dict["companies_features"]),
        (persons_nodes_with_features_df02, conf_dict["persons_features"]),
        (edges_anomalised, conf_dict["edges_features"]),
    )
    dp.write_output_path_map(output_path_map, overwrite=True)
    print("Done.")


if __name__ == "__main__":
    main()
