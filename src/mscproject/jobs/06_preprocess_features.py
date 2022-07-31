"""
- Split dataset into train, valid, and test.
- Preprocess features for modelling.
"""

from pathlib import Path

import pandas as pd
import yaml

import mscproject.dataprep as dp
import mscproject.preprocess as pre


def main() -> None:
    # Read config.
    conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

    # Load features
    print("Loading features...")
    companies_features_df = pd.read_parquet(conf_dict["companies_features"])
    persons_features_df = pd.read_parquet(conf_dict["persons_features"])
    edges_features_df = pd.read_parquet(conf_dict["edges_features"])

    # Sample records if sample size is specified.
    if conf_dict["sample_size"] is not None:
        print(f"Sampling {conf_dict['sample_size']} records...")
        companies_features_df = companies_features_df.sample(conf_dict["sample_size"])
        persons_features_df = persons_features_df.sample(conf_dict["sample_size"])
        edges_features_df = edges_features_df.sample(conf_dict["sample_size"])
    
    # Split dataset into train, valid, and test.
    print("Splitting dataset into train, valid, and test...")
    (
        train_company_nodes,
        train_person_nodes,
        train_edges,
        valid_company_nodes,
        valid_person_nodes,
        valid_edges,
        test_company_nodes,
        test_person_nodes,
        test_edges,
    ) = pre.split_datasets(
        companies_features_df, persons_features_df, edges_features_df
    )

    # Preprocess features for modelling.
    print("Preprocessing company features...")
    (
        train_company_nodes_preprocessed,
        valid_company_nodes_preprocessed,
        test_company_nodes_preprocessed,
    ) = pre.process_companies(
        train_company_nodes, valid_company_nodes, test_company_nodes
    )
    print("Preprocessing person features...")
    (
        train_person_nodes_preprocessed,
        valid_person_nodes_preprocessed,
        test_person_nodes_preprocessed,
    ) = pre.process_persons(train_person_nodes, valid_person_nodes, test_person_nodes)
    print("Preprocessing edge features...")
    (
        train_edges_preprocessed,
        valid_edges_preprocessed,
        test_edges_preprocessed,
    ) = pre.process_edges(train_edges, valid_edges, test_edges)

    # Save preprocessed features.
    def write_feature_output(
        df: pd.DataFrame,
        name: str,
        train_valid_test: str,
        preprocessed_features_path: Path,
    ) -> None:
        print(f"Saving {train_valid_test} {name}...")
        filepath = Path(
            f"{preprocessed_features_path}/{train_valid_test}/{name}.parquet"
        )
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(filepath)

    preprocessed_features_path = Path(conf_dict["preprocessed_features_path"])
    preprocessed_features_path.mkdir(parents=True, exist_ok=True)

    write_feature_output(
        train_company_nodes_preprocessed,
        "companies",
        "train",
        preprocessed_features_path,
    )
    write_feature_output(
        valid_company_nodes_preprocessed,
        "companies",
        "valid",
        preprocessed_features_path,
    )
    write_feature_output(
        test_company_nodes_preprocessed,
        "companies",
        "test",
        preprocessed_features_path,
    )
    write_feature_output(
        train_person_nodes_preprocessed,
        "persons",
        "train",
        preprocessed_features_path,
    )
    write_feature_output(
        valid_person_nodes_preprocessed,
        "persons",
        "valid",
        preprocessed_features_path,
    )
    write_feature_output(
        test_person_nodes_preprocessed,
        "persons",
        "test",
        preprocessed_features_path,
    )
    write_feature_output(
        train_edges_preprocessed,
        "edges",
        "train",
        preprocessed_features_path,
    )
    write_feature_output(
        valid_edges_preprocessed,
        "edges",
        "valid",
        preprocessed_features_path,
    )
    write_feature_output(
        test_edges_preprocessed,
        "edges",
        "test",
        preprocessed_features_path,
    )
    print("Done.")


if __name__ == "__main__":
    main()
