"""
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

    # Preprocess features for modelling.
    print("Preprocessing company features...")
    companies_preprocessed_df = pre.process_companies(companies_features_df)
    print("Preprocessing person features...")
    persons_preprocessed_df = pre.process_persons(persons_features_df)
    print("Preprocessing edge features...")
    edges_preprocessed_df = pre.process_edges(edges_features_df)

    print("Saving preprocessed features...")
    output_path_map = (
        (companies_preprocessed_df, conf_dict["companies_preprocessed"]),
        (persons_preprocessed_df, conf_dict["persons_preprocessed"]),
        (edges_preprocessed_df, conf_dict["edges_preprocessed"]),
    )
    dp.write_output_path_map(output_path_map)
    print("Done.")


if __name__ == "__main__":
    main()
