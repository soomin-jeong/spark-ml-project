
import functools

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col


spark = SparkSession.builder.appName('Delay Classifier').master('local[*]').getOrCreate()

FORBIDDEN_VARS = ["ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay",
                  "NASDelay", "SecurityDelay", "LateAircraftDelay"]
EXCLUDED_VARS = ["Year", "Origin", "Dest", "CancellationCode", "FlightNum", "TailNum"]
CATEGORICAL_VAR = ['UniqueCarrier']


class DataProcessor(object):
    def __init__(self, spark):
        self.spark = spark

    def drop_forbidden_and_excluded_variables(self, dataset):
        return dataset.drop(*(FORBIDDEN_VARS + EXCLUDED_VARS))

    def drop_duplicated_data(self, dataset):
        return dataset.drop_duplicates()

    def remove_null_arr_delay(self, dataset):
        return dataset.dropna(subset=['ArrDelay'])

    def remove_cancelled_flights(self, dataset):
        no_cancelled_flights = dataset[dataset['Cancelled'] == 0]
        no_cancelled_flights = no_cancelled_flights.drop('Cancelled')
        return no_cancelled_flights

    def split_timestring(self, dataset):
        dataset = dataset\
            .withColumn('DepHour', col('DepTime') / 100) \
            .withColumn('DepMinute', col('DepTime') % 100) \
            .withColumn('CSRDepHour', col('CRSDepTime') / 100) \
            .withColumn('CSRDepMinute', col('CRSDepTime') % 100)

        dataset = dataset.drop('DepTime', 'CRSDepTime')
        return dataset

    def convert_datatypes(self, dataset):
        def convert_to_integer(dataset, col_name):
            return dataset.withColumn(col_name, dataset[col_name].cast(IntegerType()))

        numerical_cols = dataset.schema.names
        numerical_cols.remove(*CATEGORICAL_VAR)
        print(numerical_cols)

        for each in numerical_cols:
            dataset = convert_to_integer(dataset, each)

        return dataset

    def drop_null_values(self, dataset):
        return dataset.dropna()

    def run_all_data_processing(self, dataset):
        print("[PROCESSING]: Original Schema is")
        dataset.printSchema()

        process_funcs = [self.drop_forbidden_and_excluded_variables,
                         self.remove_cancelled_flights,
                         self.drop_duplicated_data,
                         self.remove_null_arr_delay,
                         self.split_timestring,
                         self.convert_datatypes,
                         self.drop_null_values]

        for each in process_funcs:
            dataset = each(dataset)

        print("[PROCESSING]: Finished the data processing...")
        dataset.printSchema()
        return dataset
