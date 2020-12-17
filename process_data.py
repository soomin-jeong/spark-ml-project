import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.appName('Delay Classifier').master('local[*]').getOrCreate()

FORBIDDEN_VARS = ["ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay",
                  "NASDelay", "SecurityDelay", "LateAircraftDelay"]

EXCLUDED_VARS = ["Year", "Origin", "Dest", "CancellationCode"]


class DataProcessor(object):
    def drop_forbidden_and_excluded_variables(self, dataset):
        return dataset.drop(FORBIDDEN_VARS + EXCLUDED_VARS)

    def drop_duplicated_data(self, dataset):
        return dataset.dropDuplicates()

    def convert_datatypes(self, dataset):
        return dataset.withColumn('CRSDepTime', dataset['CRSDepTime'].cast(IntegerType())) \
                    .withColumn('ArrDelay', dataset['ArrDelay'].cast(IntegerType())) \
                    .withColumn('DepDelay', dataset['DepDelay'].cast(IntegerType())) \
                    .withColumn('Distance', dataset['Distance'].cast(IntegerType())) \
                    .withColumn('DayOfWeek', dataset['DayOfWeek'].cast(IntegerType())) \
                    .withColumn('Month', dataset['Month'].cast(IntegerType()))

    def remove_null_arr_delay(self, dataset):
        return dataset.dropna(subset=['ArrDelay'])

    def remove_cancelled_flights(self, dataset):
        no_cancelled_flights = dataset[dataset['Cancelled'] == 0]
        return no_cancelled_flights.pop('Cancelled')

    def convert_date_fields(self, dataset):
        if 'DepTS' in dataset and 'CSRDepTS' in dataset:
            return dataset

        temp_ts = dataset[["Year", "Month", "DayofMonth"]].astype(str).copy()

        # Actual departure time
        temp_ts['Hour'] = (dataset["DepTime"] // 100).astype(int).astype(str)
        temp_ts['Minute'] = (dataset["DepTime"] % 100).astype(int).astype(str)
        temp_ts['Time'] = temp_ts['Year'] + '-' + temp_ts['Month'] + '-' + temp_ts['DayofMonth'] + ' ' \
                          + temp_ts['Hour'] + ':' + temp_ts['Minute']
        temp_ts['DepTS'] = pd.to_datetime(temp_ts['Time'], format='%Y-%m-%d %H:%M', errors='coerce')

        # Scheduled departure time
        temp_ts['CSRDepHour'] = (dataset["CRSDepTime"] // 100).astype(int).astype(str)
        temp_ts['CSRDepMinute'] = (dataset["CRSDepTime"] % 100).astype(int).astype(str)
        temp_ts['CSRTime'] = temp_ts['Year'] + '-' + temp_ts['Month'] + '-' + temp_ts['DayofMonth'] + ' ' + \
                             temp_ts['CSRDepHour'] + ':' + temp_ts['CSRDepMinute']
        temp_ts['CSRDepTS'] = pd.to_datetime(temp_ts['CSRTime'], format='%Y-%m-%d %H:%M', errors='coerce')

        dataset['DepTS'] = temp_ts['DepTS']
        dataset['CSRDepTS'] = temp_ts['CSRDepTS']
        return dataset

    def run_all_data_processing(self, dataset):
        dataset = self.drop_forbidden_and_excluded_variables(dataset)
        dataset = self.remove_cancelled_flights(dataset)
        dataset = self.drop_duplicated_data(dataset)
        dataset = self.convert_datatypes(dataset)
        dataset = self.remove_null_arr_delay(dataset)
        dataset = self.convert_datatypes(dataset)
        return dataset

data_processor = DataProcessor()