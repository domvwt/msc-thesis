from pathlib import Path

import pandas as pd
import torch
import yaml

import mscproject.pygloaders as pgl


def main():

    # Read config.
    conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

    companies_df = pd.read_parquet(conf_dict["companies_preprocessed"])
    persons_df = pd.read_parquet(conf_dict["persons_preprocessed"])
    edges_df = pd.read_parquet(conf_dict["edges_preprocessed"])

    # Ensure no self loops!
    edges_df = edges_df[edges_df["src"] != edges_df["dst"]]

    print("Loading PyTorch Geometric data...")
    pyg_data = pgl.graph_elements_to_heterodata(companies_df, persons_df, edges_df)

    print("Saving PyTorch Geometric data...")
    pyg_data_path = Path(conf_dict["pyg_data"])
    pyg_data_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pyg_data, pyg_data_path)

    print("Data schema:")
    print(pyg_data)

    print("Done.")


if __name__ == "__main__":
    main()
