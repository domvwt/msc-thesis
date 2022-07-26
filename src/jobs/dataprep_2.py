from pathlib import Path

import yaml
from pyspark.sql import SparkSession

import src.dataprep as dp


def main() -> None:
    # Read config.
    conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

    # Start session.
    spark = SparkSession.builder.getOrCreate()

    # Load interim data.
    companies_interim_df = spark.read.parquet(conf_dict["companies_interim"])
    relationships_interim_df = spark.read.parquet(conf_dict["relationships_interim"])
    persons_interim_df = spark.read.parquet(conf_dict["persons_interim"])
    companies_house_interim_df = spark.read.parquet(
        conf_dict["companies_house_interim"]
    )

    # Process interim data.
    processed_companies_df = dp.process_companies(
        companies_interim_df, companies_house_interim_df
    )
    processed_relationships_df = dp.process_relationships(relationships_interim_df)
    processed_persons_df = dp.process_persons(persons_interim_df)
    processed_addresses_df = dp.process_addresses(
        companies_interim_df, persons_interim_df
    )

    # Convert statementIDs.
    processed_companies_df = dp.strip_chars_from_statement_id(processed_companies_df)
    processed_relationships_df = dp.strip_chars_from_statement_id(
        processed_relationships_df, "interestedPartyStatementID"
    )
    processed_relationships_df = dp.strip_chars_from_statement_id(
        processed_relationships_df, "subjectStatementID"
    )
    processed_persons_df = dp.strip_chars_from_statement_id(processed_persons_df)
    processed_addresses_df = dp.strip_chars_from_statement_id(processed_addresses_df)

    # Write outputs.
    output_path_map = {
        processed_companies_df: conf_dict["companies_processed"],
        processed_relationships_df: conf_dict["relationships_processed"],
        processed_persons_df: conf_dict["persons_processed"],
        processed_addresses_df: conf_dict["addresses_processed"],
    }
    dp.write_if_missing(output_path_map)

    # Stop session.
    spark.stop()


if __name__ == "__main__":
    main()
