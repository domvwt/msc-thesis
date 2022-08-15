from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import torch
import joblib as jl
import yaml
from torch_geometric.data import HeteroData

import mscproject.pygloaders as pgl

def main():

    # Read config.
    conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

    features_root_path = Path(conf_dict["preprocessed_features_path"])
    data_splits = ["train", "valid", "test"]

    for data_split in data_splits:
        data_split_input_path = features_root_path / data_split
        data_split_output_path = Path(conf_dict["pyg_data_path"]) / f"{data_split}.pt"
        print(f"Loading {data_split} from {data_split_input_path}")
        pyg_data = pgl.load_data_to_pyg(
            data_split_input_path / "companies.parquet",
            data_split_input_path / "persons.parquet",
            data_split_input_path / "edges.parquet",
        )
        print(f"Saving {data_split} to {data_split_output_path}")
        data_split_output_path.parent.mkdir(parents=True, exist_ok=True)
        jl.dump(pyg_data, data_split_output_path)


if __name__ == "__main__":
    main()
