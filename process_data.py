
import functools

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import count, col, isnull, when, isnan


spark = SparkSession.builder.appName('Delay Classifier').master('local[*]').getOrCreate()

FORBIDDEN_VARS = ["ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay",
                  "NASDelay", "SecurityDelay", "LateAircraftDelay"]

EXCLUDED_VARS = ["Year", "Origin", "Dest", "CancellationCode", "FlightNum", "TailNum"]


class DataProcessor(object):
    def __init__(self, spark):
        self.spark = spark

        self.nominal_category_vars = ["UniqueCarrier", "FlightNum", "TailNum", "Cancelled"]
        self.ordinal_category_vars = ["Year", "DayOfWeek"]
        self.numerical_vars = ["CRSElapsedTime", "ArrDelay", "DepDelay", "Distance", "TaxiOut", "DayofMonth", "Month"]
        self.string_vars = ["DepTime", "CRSDepTime", "CRSArrTime", "Origin", "Dest", "CancellationCode"]
        # [NOTE] DayofMonth, Month : though they are ordial category, since there are too many categories (up to 12 and 31),
        # we regard them as numerical vars

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

    #  Split the timestring(hhmm) into hour and minute and add them to the dataset
    def split_timestring(self, dataset):
        dataset = dataset\
            .withColumn('DepHour', col('DepTime') / 100) \
            .withColumn('DepMinute', col('DepTime') % 100) \
            .withColumn('CRSDepHour', col('CRSDepTime') / 100) \
            .withColumn('CRSDepMinute', col('CRSDepTime') % 100) \
            .withColumn('CRSArrHour', col('CRSArrTime') / 100) \
            .withColumn('CRSArrMinute', col('CRSArrTime') % 100)

        # Adding the newly added numerical variables into the list of numerical vars
        self.numerical_vars = self.numerical_vars + ['DepHour', 'DepMinute', 'CRSDepHour', 'CRSDepMinute',
                                                     'CRSArrHour', 'CRSArrMinute']
        # Removing the unnecessary timestring(HHmm) fields
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

    def drop_too_many_null_values(self, dataset):
        threshold = 0.8 * dataset.count()
        null_counts = dataset.select([count(when(col(c).isNull(), c)).alias(c) for c in dataset.columns]).collect()[
            0].asDict()
        to_drop = [k for k, v in null_counts.items() if v > threshold]
        return dataset.drop(*to_drop)

    def run_all_data_processing(self, dataset):
        process_func = [self.drop_forbidden_and_excluded_variables,
                        self.remove_cancelled_flights,
                        self.drop_duplicated_data,
                        self.remove_null_arr_delay,
                        self.split_timestring,
                        self.convert_datatypes,
                        self.drop_too_many_null_values,
                        self.drop_null_values]

        for each in process_func:
            dataset = each(dataset)

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
