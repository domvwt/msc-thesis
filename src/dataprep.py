from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def write_if_missing(outputs_map: dict[DataFrame, str]) -> None:
    for dataframe, path in outputs_map.items():
        if not Path(path).exists():
            dataframe.write.parquet(path)


def strip_chars_from_statement_id(df: DataFrame, id_col: str = "statementID") -> DataFrame:
    return df.withColumn(
        id_col, F.col(id_col).substr(F.lit(24), F.length(id_col))
    )


def extract_companies(ownership_df: DataFrame) -> DataFrame:
    companies_filter = (
        (F.col("incorporatedInJurisdiction.code") == "GB")
        & (F.col("statementType") == "entityStatement")
        & (F.col("dissolutionDate").isNull())
    )
    gb_companies_df = ownership_df.filter(companies_filter)
    gb_companies_df.repartition(100)
    return gb_companies_df


def extract_relationships(
    ownership_df: DataFrame, companies_df: DataFrame
) -> DataFrame:
    relationships_df = ownership_df.filter(
        F.col("statementType") == "ownershipOrControlStatement"
    )
    join_expr = F.col("relationship.subject.describedByEntityStatement") == F.col(
        "company.statementID"
    )
    gb_relationships_df = relationships_df.alias("relationship").join(
        companies_df.alias("company"), join_expr, "leftsemi"
    )
    gb_relationships_df.repartition(100)
    return gb_relationships_df


def extract_persons(ownership_df: DataFrame, relationships_df: DataFrame) -> DataFrame:
    persons_df = ownership_df.filter(F.col("statementType") == "personStatement")
    join_expr = F.col("person.statementID") == F.col(
        "relationship.interestedParty.describedByPersonStatement"
    )
    gb_persons_df = persons_df.alias("person").join(
        relationships_df.alias("relationship"), join_expr, "leftsemi"
    )
    gb_persons_df.repartition(100)
    return gb_persons_df


def process_companies(companies_df: DataFrame) -> DataFrame:
    """Select and flatten relevant company information.

    Args:
        companies_df (DataFrame): Unprocessed company records.

    Returns:
        DataFrame: Processed company records.
    """
    return companies_df.select("statementID", "name", "foundingDate")


def process_relationships(relationships_df: DataFrame) -> DataFrame:
    """Select and flatten relevant information from ownership statements.

    Args:
        relationships_df (DataFrame): Unprocessed ownership statements.

    Returns:
        DataFrame: Flattened and processed ownership statements.
    """
    return (
        relationships_df.withColumn(
            "interestedPartyStatementID",
            F.coalesce(
                "interestedParty.describedByEntityStatement",
                "interestedParty.describedByPersonStatement",
            ),
        )
        .withColumn(
            "interestedPartyIsPerson",
            F.col("interestedParty.describedByPersonStatement").isNotNull(),
        )
        .withColumn("subjectStatementID", F.col("subject.describedByEntityStatement"))
        .select("*", F.explode("interests").alias("interestsExploded"))
        .select("*", "interestsExploded.*")
        .select("*", F.col("share.minimum").alias("minimumShare"))
        .filter(F.col("startDate").isNotNull())
        .filter(F.col("endDate").isNull())
        .filter(F.col("type") == "shareholding")
        .filter(F.col("details").endswith("percent"))
        .groupBy(
            "interestedPartyStatementID",
            "interestedPartyIsPerson",
            "subjectStatementID",
        )
        .agg(F.max("minimumShare").alias("minimumShare"))
    )


def process_persons(persons_df: DataFrame) -> DataFrame:
    """Select and flatten relevant person information.

    Args:
        persons_df (DataFrame): Unprocessed person records.

    Returns:
        DataFrame: Processed person records.
    """
    return (
        persons_df.select("*", F.explode("nationalities"))
        .select("*", F.col("col.code").alias("nationality"))
        .groupBy("statementID", "birthDate")
        .agg(F.max("nationality").alias("nationality"))
    )


def process_addresses(companies_df: DataFrame, persons_df: DataFrame) -> DataFrame:
    """Select and flatten relevant address information.

    Args:
        companies_df (DataFrame): Unprocessed company records.
        persons_df (DataFrame): Unprocessed person records.

    Returns:
        DataFrame: Addresses mapped to person and company entities.
    """
    entities_df = companies_df.unionByName(persons_df)
    entity_address_edge_df = entities_df.select(
        "statementID", F.explode("addresses")
    ).select("statementID", "col.address")
    return entity_address_edge_df
