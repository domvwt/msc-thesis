from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import yaml
import joblib as jl
from torch_geometric.data.in_memory_dataset import InMemoryDataset

import mscproject.pygloaders_v2 as pgl


def main():

    # Read config.
    conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

    companies_df = pd.read_parquet(conf_dict["companies_preprocessed"])
    persons_df = pd.read_parquet(conf_dict["persons_preprocessed"])
    edges_df = pd.read_parquet(conf_dict["edges_preprocessed"])

    # Ensure no self loops!
    edges_df = edges_df[edges_df["src"] != edges_df["dst"]]

    print("Processing data for PyTorch Geometric...")
    pyg_data_list = pgl.graph_elements_to_heterodata(companies_df, persons_df, edges_df)

    print("Saving PyTorch Geometric data...")
    pyg_data_dir = Path(conf_dict["pyg_data"])
    pyg_data_dir.mkdir(parents=True, exist_ok=True)

    # Save each element of the list in parallel using joblib.
    jl.Parallel(n_jobs=16, prefer="threads")(
        jl.delayed(torch.save)(pyg_data, pyg_data_dir / f"{i}.pt")
        for i, pyg_data in enumerate(pyg_data_list)
    )

    print("Data schema:")
    print(pyg_data_list)

    print("Done.")


if __name__ == "__main__":
    main()
