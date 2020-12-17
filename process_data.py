import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.appName('Delay Classifier').master('local[*]').getOrCreate()

FORBIDDEN_VARS = ["ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]


class DataProcessor(object):
    def drop_forbidden_variables(self, dataset):
        return dataset.drop(FORBIDDEN_VARS, axis=1)

    def drop_duplicated_data(self, dataset):
        return dataset.drop_duplicates()

    def remove_null_arr_delay(self, dataset):
        return dataset.dropna(subset=['ArrDelay'])

    def remove_cancelled_flights(self, dataset):
        no_cancelled_flights = dataset[dataset['Cancelled'] == 0]
        no_cancelled_flights.pop('Cancelled')
        return no_cancelled_flights

    def remove_cancelleation_code(self, dataset):
        dataset.pop("CancellationCode")
        return dataset

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

        # dataset['DepTS'] = temp_ts['DepTS']
        dataset['DepHour'] = temp_ts['Hour']
        dataset['DepMinute'] = temp_ts['Minute']

        # dataset['CSRDepTS'] = temp_ts['CSRDepTS']
        dataset['CSRDepHour'] = temp_ts['CSRDepHour']
        dataset['CSRDepMinute'] = temp_ts['CSRDepMinute']
        return dataset

    def run_all_data_processing(self, dataset):
        dataset = self.drop_duplicated_data(dataset)
        dataset = self.drop_forbidden_variables(dataset)
        dataset = self.remove_null_arr_delay(dataset)
        dataset = self.remove_cancelled_flights(dataset)
        dataset = self.remove_cancelleation_code(dataset)
        dataset = self.convert_date_fields(dataset)
        return dataset

data_processor = DataProcessor()