from pathlib import Path

import yaml
from pyspark.sql import SparkSession

import src.dataprep as dp


def main() -> None:
    # Read config.
    conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

    # Start session
    spark = SparkSession.builder.config("spark.driver.memory", "8g").getOrCreate()

    # Extract entities.
    ownership_data_path = Path(conf_dict["ownership_data_raw"]).absolute()
    ownership_df = spark.read.json(f"file://{ownership_data_path}")
    relationships_df = dp.extract_relationships(ownership_df)
    companies_df = dp.extract_companies(ownership_df)
    persons_df = dp.extract_persons(ownership_df, relationships_df)
    companies_house_df = dp.extract_companies_house_info(
        spark, conf_dict["companies_house_data_raw"]
    )

    # Write outputs.
    output_path_map = {
        companies_df: conf_dict["companies_interim"],
        relationships_df: conf_dict["relationships_interim"],
        persons_df: conf_dict["persons_interim"],
        companies_house_df: conf_dict["companies_house_interim"],
    }
    dp.write_if_missing(output_path_map)


if __name__ == "__main__":
    main()
