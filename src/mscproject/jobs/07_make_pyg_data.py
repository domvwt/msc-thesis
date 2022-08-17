from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import yaml

import mscproject.pygloaders as pgl
from torch_geometric.data.in_memory_dataset import InMemoryDataset


def main():

    # Read config.
    conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

    companies_df = pd.read_parquet(conf_dict["companies_preprocessed"])
    persons_df = pd.read_parquet(conf_dict["persons_preprocessed"])
    edges_df = pd.read_parquet(conf_dict["edges_preprocessed"])

    print("Loading PyTorch Geometric data...")
    pyg_data = pgl.graph_elements_to_heterodata(companies_df, persons_df, edges_df)

    print("Saving PyTorch Geometric data...")
    torch.save(pyg_data, conf_dict["pyg_data"])

    print("Done.")

if __name__ == "__main__":
    main()
