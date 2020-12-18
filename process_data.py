
import functools

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col


spark = SparkSession.builder.appName('Delay Classifier').master('local[*]').getOrCreate()

FORBIDDEN_VARS = ["ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay",
                  "NASDelay", "SecurityDelay", "LateAircraftDelay"]

# TODO: UniqueCarrier Makes an error on DT, to test remove "UniqueCarrier" from the list below.
EXCLUDED_VARS = ["Year", "Origin", "Dest", "CancellationCode", "FlightNum", "TailNum", "UniqueCarrier"]


class DataProcessor(object):
    def __init__(self, spark):
        self.spark = spark

        # TODO: Are the variables analyzed correctly? Exceptions: DayofMonth, Month
        self.nominal_category_vars = ["UniqueCarrier", "FlightNum", "TailNum", "Cancelled"]
        self.ordinal_category_vars = ["Year", "DayOfWeek"]
        self.numerical_vars = ["CRSElapsedTime", "ArrDelay", "DepDelay", "Distance", "TaxiOut", "DayofMonth", "Month"]
        self.string_vars = ["DepTime", "CRSDepTime", "ArrTime", "CRSArrTime", "Origin", "Dest", "CancellationCode"]
        # DayofMonth, Month : though they are ordial category, since there are too many categories (up to 12 and 31),
        #              we moved it to numerical vars

    def remove_from_list(self, list_to_check, list_to_survive):
        to_remove = []
        for each in list_to_check:
            if each not in list_to_survive:
                to_remove.append(each)
        for each in to_remove:
            list_to_check.remove(each)

    def update_post_proessing_vars(self, post_process_vars):
        for each_var_list in [self.nominal_category_vars, self.ordinal_category_vars, self.numerical_vars, self.string_vars]:
            self.remove_from_list(each_var_list, post_process_vars)

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
            .withColumn('CRSDepHour', col('CRSDepTime') / 100) \
            .withColumn('CRSDepMinute', col('CRSDepTime') % 100) \
            .withColumn('CRSArrHour', col('CRSArrTime') / 100) \
            .withColumn('CRSArrMinute', col('CRSArrTime') % 100)

        self.numerical_vars = self.numerical_vars + ['DepHour', 'DepMinute', 'CRSDepHour', 'CRSDepMinute',
                                                     'CRSArrHour', 'CRSArrMinute']

        dataset = dataset.drop('DepTime', 'CRSDepTime', 'CRSArrTime')
        return dataset

    def convert_datatypes(self, dataset):
        def convert_to_integer(dataset, col_name):
            return dataset.withColumn(col_name, dataset[col_name].cast(IntegerType()))

        for each in self.numerical_vars:
            dataset = convert_to_integer(dataset, each)

        return dataset

    def drop_null_values(self, dataset):
        return dataset.dropna()

    def run_all_data_processing(self, dataset):
        print("[PROCESSING]: Original Schema")
        dataset.printSchema()
        print("================================")

        process_funcs = [self.drop_forbidden_and_excluded_variables,
                         self.remove_cancelled_flights,
                         self.drop_duplicated_data,
                         self.remove_null_arr_delay,
                         self.split_timestring,
                         self.convert_datatypes,
                         self.drop_null_values]

        for each in process_funcs:
            dataset = each(dataset)

        print("[PROCESSING]: New Schema")
        dataset.printSchema()
        print("================================")

        self.update_post_proessing_vars(dataset.schema.names)
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
