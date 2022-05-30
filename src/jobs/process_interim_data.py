from pathlib import Path

import yaml
from pyspark.sql import SparkSession

import src.dataprep as dp


def main() -> None:
    # Read config.
    conf_dict = yaml.safe_load(Path("config/dataprep.yaml").read_text())

    companies_interim_path = str(Path(conf_dict["companies_interim"]).absolute())
    relationships_interim_path = str(Path(conf_dict["relationships_interim"]).absolute())
    persons_interim_path = str(Path(conf_dict["persons_interim"]).absolute())

    companies_processed_path = str(Path(conf_dict["companies_processed"]).absolute())
    relationships_processed_path = str(Path(conf_dict["relationships_processed"]).absolute())
    persons_processed_path = str(Path(conf_dict["persons_processed"]).absolute())
    addresses_processed_path = str(Path(conf_dict["addresses_processed"]).absolute())

    # Start session.
    spark = SparkSession.builder.getOrCreate()

    # Load interim data.
    companies_interim_df = spark.read.parquet(companies_interim_path)
    relationships_interim_df = spark.read.parquet(relationships_interim_path)
    persons_interim_df = spark.read.parquet(persons_interim_path)

    # Process interim data.
    processed_companies_df = dp.process_companies(companies_interim_df)
    processed_relationships_df = dp.process_relationships(relationships_interim_df)
    processed_persons_df = dp.process_persons(persons_interim_df)
    processed_addresses_df = dp.process_addresses(companies_interim_df, persons_interim_df)

    # Convert statementIDs.
    processed_companies_df = dp.strip_chars_from_statement_id(processed_companies_df)
    processed_relationships_df = dp.strip_chars_from_statement_id(processed_relationships_df, "interestedPartyStatementID")
    processed_relationships_df = dp.strip_chars_from_statement_id(processed_relationships_df, "subjectStatementID")
    processed_persons_df = dp.strip_chars_from_statement_id(processed_persons_df)
    processed_addresses_df = dp.strip_chars_from_statement_id(processed_addresses_df)

    # Write outputs.
    output_path_map = {
        processed_companies_df: companies_processed_path,
        processed_relationships_df: relationships_processed_path,
        processed_persons_df: persons_processed_path,
        processed_addresses_df: addresses_processed_path
    }
    dp.write_if_missing(output_path_map)


if __name__ == "__main__":
    main()
