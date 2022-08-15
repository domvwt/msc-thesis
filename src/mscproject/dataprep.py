import itertools as it
import shutil
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession


def write_if_missing_spark(outputs_map: dict[DataFrame, str]) -> None:
    for dataframe, path in outputs_map.items():
        if not Path(path).exists():
            dataframe.write.parquet(path)


def write_output_path_map(
    outputs_map: Iterable[Tuple[pd.DataFrame, str]], overwrite: bool = False
) -> None:
    for df, path in outputs_map:
        path_object = Path(path)
        if not Path(path).exists() or overwrite:
            path_object.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)


def strip_chars_from_statement_id(
    df: DataFrame, id_col: str = "statementID"
) -> DataFrame:
    return df.withColumn(id_col, F.col(id_col).substr(F.lit(24), F.length(id_col)))


def extract_relationships(ownership_df: DataFrame) -> DataFrame:
    """Extract all ownership statements pertaining to GB companies.

    Args:
        ownership_df (DataFrame): Raw open ownership data.

    Returns:
        DataFrame: Ownership relationships to GB companies.
    """
    relationships_df = ownership_df.filter(
        F.col("statementType") == "ownershipOrControlStatement"
    )
    companies_filter = (
        (F.col("incorporatedInJurisdiction.code") == "GB")
        & (F.col("statementType") == "entityStatement")
        & (F.col("dissolutionDate").isNull())
    )
    gb_company_ids = ownership_df.filter(companies_filter)
    join_expr = F.col("relationship.subject.describedByEntityStatement") == F.col(
        "company.statementID"
    )
    gb_relationships_df = relationships_df.alias("relationship").join(
        gb_company_ids.alias("company"), join_expr, "leftsemi"
    )
    gb_relationships_df.repartition(100)
    return gb_relationships_df


def extract_companies(ownership_df: DataFrame) -> DataFrame:
    # Restricted to GB companies due to resource constraints
    companies_filter = (F.col("statementType") == "entityStatement") & (
        F.col("incorporatedInJurisdiction.code") == "GB"
    )
    companies_df = ownership_df.filter(companies_filter)
    companies_df.repartition(100)
    return companies_df


def extract_persons(ownership_df: DataFrame, relationships_df: DataFrame) -> DataFrame:
    persons_df = ownership_df.filter(F.col("statementType") == "personStatement")
    join_expr = F.col("person.statementID") == F.col(
        "relationship.interestedParty.describedByPersonStatement"
    )
    relevant_persons_df = persons_df.alias("person").join(
        relationships_df.alias("relationship"), join_expr, "leftsemi"
    )
    relevant_persons_df.repartition(100)
    return relevant_persons_df


def extract_companies_house_info(spark: SparkSession, csv_path: str) -> DataFrame:
    # Remove invalid characters (<space>, ',') from column names
    with Path(csv_path).open() as f:
        column_names = f.readline().split(",")
    column_names_clean = [s.strip().replace(".", "_") for s in column_names]
    df00 = spark.read.csv(csv_path, header=False)
    cols_renamed = [
        F.col(colname).alias(colname_new)
        for colname, colname_new in zip(df00.columns, column_names_clean)
    ]
    # Select relevant columns
    df01 = df00.select(cols_renamed).filter(F.col("CompanyName") != "CompanyName")
    keep_cols = [
        "CompanyName",
        "CompanyNumber",
        "CompanyCategory",
        "CompanyStatus",
        "Accounts_AccountCategory",
        "SICCode_SicText_1",
    ]
    df02 = df01.select(keep_cols)
    return df02


def process_companies(
    companies_df: DataFrame, companies_house_df: DataFrame
) -> DataFrame:
    """Select and flatten relevant company information.

    Args:
        companies_df (DataFrame): Unprocessed company records.
        companies_house_df (DataFrame): Companies House records.

    Returns:
        DataFrame: Processed company records.
    """
    # Extract relevant scheme IDs from company records.
    scheme_dict = {
        "OpenOwnership Register": "openOwnershipRegisterID",
        "Companies House": "companiesHouseID",
        "GB Persons Of Significant Control Register - Registration numbers": "companiesHouseID",
        "OpenCorporates": "openCorporatesID",
    }
    scheme_map = F.create_map(*(F.lit(item) for item in it.chain(*scheme_dict.items())))
    scheme_ids_long = companies_df.select(
        "statementID", F.explode("identifiers")
    ).select("statementID", "col.id", "col.schemeName")
    scheme_ids_wide = (
        scheme_ids_long.filter(F.col("schemeName").isin(list(scheme_dict.keys())))
        .withColumn("schemeName", scheme_map[F.col("schemeName")])
        .groupBy("statementID")
        .pivot("schemeName")
        .agg(F.max("id"))
    )
    # Join scheme IDs to companies data.
    output_df = companies_df.select(
        "statementID",
        "name",
        "foundingDate",
        "dissolutionDate",
        F.col("incorporatedInJurisdiction.code").alias("countryCode"),
    )
    output_df = output_df.join(scheme_ids_wide, on=["statementID"], how="left")
    # Join Companies House info to companies data.
    join_expr = F.col("CompaniesHouseID") == F.col("CompanyNumber")
    output_df = output_df.join(companies_house_df, on=join_expr, how="left")
    # Drop unneeded columns.
    drop_cols = ["CompanyName", "CompanyNumber"]
    output_df = output_df.drop(*drop_cols)
    return output_df


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
    return persons_df.select(
        "statementID",
        "birthDate",
        F.col("names").getItem(0).getItem("fullName").alias("name"),
        F.col("nationalities").getItem(0).getItem("code").alias("nationality"),
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
        "statementID", F.col("addresses").getItem(0).alias("address")
    )
    return entity_address_edge_df


def write_csv(df: DataFrame, output_dir: str, output_filename: str) -> None:
    csv_path = Path(output_dir) / f"{output_filename}"
    parquet_csv_path = f"{csv_path}.parquet"

    df.coalesce(1).write.csv(parquet_csv_path, header=True, mode="overwrite")
    csv_txt_path = next(Path.cwd().glob(f"{parquet_csv_path}/part-00000*.csv"))
    Path(csv_txt_path).rename(csv_path)
    shutil.rmtree(parquet_csv_path)
