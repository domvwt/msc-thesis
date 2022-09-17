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

    # Get new component ids.
    print("Updating component ids...")
    new_graph = feat.make_graph(edges_anomalised)
    node_id_to_cc_id_map = feat.get_node_id_to_cc_id_map(new_graph)
    companies_anomalised["component"] = companies_anomalised["id"].map(
        node_id_to_cc_id_map
    )
    persons_anomalised["component"] = persons_anomalised["id"].map(node_id_to_cc_id_map)
    edges_df["component"] = edges_df["src"].map(node_id_to_cc_id_map)

    companies_anomalised = pd.read_parquet(conf_dict["companies_anomalies"])
    persons_anomalised = pd.read_parquet(conf_dict["persons_anomalies"])
    edges_anomalised = pd.read_parquet(conf_dict["edges_anomalies"])

    # Drop components with < 9 edges (i.e. 10 nodes).
    print("Dropping components with < 9 edges...")
    (
        edges_anomalised,
        dropped_component_set,
    ) = feat.drop_components_with_less_than_n_edges(edges_anomalised, 9)
    print(f"Dropped {len(dropped_component_set)} components...")

    # Drop nodes with no edges.
    print("Dropping nodes with no edges...")
    companies_anomalised = companies_anomalised.query(
        "component not in @dropped_component_set"
    )
    persons_anomalised = persons_anomalised.query(
        "component not in @dropped_component_set"
    )

    # Save features.
    print("Saving anomalies...")
    # ? Swapped dict for list of tuples as pd.DataFrame is unhashable.
    output_path_map = [
        (companies_anomalised, conf_dict["companies_anomalies"]),
        (persons_anomalised, conf_dict["persons_anomalies"]),
        (edges_anomalised, conf_dict["edges_anomalies"]),
    ]
    dp.write_output_path_map(output_path_map)
    print("Done.")


if __name__ == "__main__":
    main()
