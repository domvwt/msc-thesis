import sys
from pathlib import Path

from pyspark.sql import SparkSession

import src.dataprep as dp


def main():
    source_parquet_path = Path(sys.argv[1])
    output_filename = source_parquet_path.stem + ".csv"

    spark = SparkSession.builder.getOrCreate()

    df = spark.read.parquet(str(source_parquet_path))
    dp.write_csv(df, str(source_parquet_path.parent), output_filename)


if __name__ == "__main__":
    main()
