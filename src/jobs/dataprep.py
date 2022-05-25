from pathlib import Path

import pyspark.sql.functions as F
import yaml
from pyspark.sql import SparkSession


def main() -> None:
    conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())
    ownership_data_raw = str(Path(conf_dict["ownership_data_raw"]).absolute())
    companies_interim = str(Path(conf_dict["companies_interim"]).absolute())
    relationships_interim = str(Path(conf_dict["relationships_interim"]).absolute())
    persons_interim = str(Path(conf_dict["persons_interim"]).absolute())

    spark = SparkSession.builder.getOrCreate()

    df = spark.read.json(f"file://{ownership_data_raw}")

    companies_filter = (F.col("incorporatedInJurisdiction.code") == "GB") & (
        F.col("statementType") == "entityStatement"
    )
    gb_companies_df = df.filter(companies_filter)
    gb_companies_df.repartition(200)
    gb_companies_df.write.parquet(companies_interim)

    relationships_df = df.filter(
        F.col("statementType") == "ownershipOrControlStatement"
    )
    join_expr = F.col("relationship.subject.describedByEntityStatement") == F.col(
        "company.statementID"
    )
    gb_relationships_df = relationships_df.alias("relationship").join(
        gb_companies_df.alias("company"), join_expr, "leftsemi"
    )
    gb_relationships_df.repartition(200)
    gb_relationships_df.write.parquet(relationships_interim)

    persons_df = df.filter(F.col("statementType") == "personStatement")
    join_expr = F.col("person.statementID") == F.col(
        "relationship.interestedParty.describedByPersonStatement"
    )
    gb_persons_df = persons_df.alias("person").join(
        gb_relationships_df.alias("relationship"), join_expr, "leftsemi"
    )
    gb_persons_df.repartition(200)
    gb_persons_df.write.parquet(persons_interim)


if __name__ == "__main__":
    main()
