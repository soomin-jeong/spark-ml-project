
import functools

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col, floor, count, when


FORBIDDEN_VARS = ["ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay",
                  "NASDelay", "SecurityDelay", "LateAircraftDelay"]

NUMERICAL_VARS = ["CRSDepTime", "CRSArrTime", "DepTime", "ArrDelay", "DepDelay", "Distance", "TaxiOut", "Cancelled",
                  "Year", "DayOfWeek", "DayofMonth", "Month"]

EXCLUDED_VARS = ["Year", "Origin", "Dest", "FlightNum", "TailNum", "CancellationCode", "Cancelled", "CRSElapsedTime",
                 "CRSDepTime", "Distance", "DayofMonth", "Month", "DayOfWeek"]


class DataProcessor(object):
    def __init__(self, spark):
        self.spark = spark

    def drop_too_many_null_values(self, dataset):
        # Function to stop model from functioning on old datasets
        threshold = 0.8 * dataset.count()
        null_counts = dataset.select([count(when(col(c).isNull(), c)).alias(c) for c in dataset.columns]).collect()[
            0].asDict()
        to_drop = [k for k, v in null_counts.items() if v > threshold]
        return dataset.drop(*to_drop)

    def perform_data_cleaning(self, dataset):
        # Delete the forbidden columns
        dataset = dataset.drop(*FORBIDDEN_VARS)

        #Drop duplicates
        dataset = dataset.drop_duplicates()

        # Convert to right
        for var in NUMERICAL_VARS:
            dataset = dataset.withColumn(var, dataset[var].cast(IntegerType()))

        # Filter on cancelled flights
        dataset = dataset.where("Cancelled == 0")

        # Drop irrelevant features
        dataset = dataset.drop(*EXCLUDED_VARS)

        dataset = dataset.na.fill({'TaxiOut': 0})

        # Filter all the rows where ArrDelay is unknown
        dataset = dataset.where(dataset.ArrDelay.isNotNull())
        dataset.printSchema()
        #dataset = self.drop_too_many_null_values(dataset) # slow
        return dataset

    def feature_engineering(self, dataset):
        # Exract hour of Departure time and planned arrival time
        dataset = dataset.withColumn('ProcDepTime', floor(col('DepTime') / 100)) \
            .withColumn('ProcArrTime', floor(col('CRSArrTime') / 100))
        return dataset

    def run_all_data_processing(self, dataset):
        dataset = self.perform_data_cleaning(dataset)
        dataset = self.feature_engineering(dataset)
        print("[PROCESS] Finished processing the raw dataset")
        return dataset

    def string_indexer(self, input_cols):
        output_cols = [each_col + "_idx" for each_col in input_cols]
        return StringIndexer(inputCols=input_cols, outputCols=output_cols), output_cols

    def one_hot_encoder(self, input_cols):
        output_cols = [each_col + "_vec" for each_col in input_cols]
        return OneHotEncoder(inputCols=input_cols, outputCols=output_cols), output_cols

    def vector_assembler(self, input_cols):
        # following the default name, "features" for models
        return VectorAssembler(inputCols=input_cols, outputCol='features')
