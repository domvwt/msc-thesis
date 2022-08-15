from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml

import mscproject.pygloaders as pgl


def main():

    # Read config.
    conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())
    data_splits = ["train", "valid", "test"]

    for data_split in data_splits:
        data_split_input_dir = (
            Path(conf_dict["preprocessed_features_path"]) / data_split
        )
        data_split_output_dir = Path(conf_dict["pyg_data_path"]) / data_split
        print(f"Loading {data_split} data from {data_split_input_dir}")
        data_list = pgl.load_data_to_pyg(
            data_split_input_dir / "companies.parquet",
            data_split_input_dir / "persons.parquet",
            data_split_input_dir / "edges.parquet",
        )
        print(f"Saving {data_split} to {data_split_output_dir}")
        data_split_output_dir.mkdir(parents=True, exist_ok=True)
        for component_id, data in enumerate(data_list):
            torch.save(data, data_split_output_dir / f"{component_id}.pt")


if __name__ == "__main__":
    main()
