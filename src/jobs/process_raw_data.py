from pathlib import Path

import yaml
from pyspark.sql import SparkSession

import src.dataprep as dp


def main() -> None:
    # Read config.
    conf_dict = yaml.safe_load(Path("config/dataprep.yaml").read_text())
    ownership_data_raw = str(Path(conf_dict["ownership_data_raw"]).absolute())
    companies_house_data_raw = str(
        Path(conf_dict["companies_house_data_raw"]).absolute()
    )
    companies_interim_path = str(Path(conf_dict["companies_interim"]).absolute())
    relationships_interim_path = str(
        Path(conf_dict["relationships_interim"]).absolute()
    )
    persons_interim_path = str(Path(conf_dict["persons_interim"]).absolute())
    companies_house_interim_path = str(
        Path(conf_dict["companies_house_interim"]).absolute()
    )

    # Start session.
    spark = SparkSession.builder.config("spark.driver.memory", "8g").getOrCreate()
    ownership_df = spark.read.json(f"file://{ownership_data_raw}")

    # Extract entities.
    relationships_df = dp.extract_relationships(ownership_df)
    companies_df = dp.extract_companies(ownership_df, relationships_df)
    persons_df = dp.extract_persons(ownership_df, relationships_df)
    companies_house_df = dp.extract_companies_house_info(
        spark, companies_house_data_raw
    )

    # Write outputs.
    output_path_map = {
        companies_df: companies_interim_path,
        relationships_df: relationships_interim_path,
        persons_df: persons_interim_path,
        companies_house_df: companies_house_interim_path,
    }
    dp.write_if_missing(output_path_map)


if __name__ == "__main__":
    main()
