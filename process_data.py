
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Delay Classifier').master('local[*]').getOrCreate()

FORBIDDEN_VARS = ["ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay",
                  "NASDelay", "SecurityDelay", "LateAircraftDelay"]
EXCLUDED_VARS = ["Year", "Origin", "Dest", "CancellationCode", "FlightNum", "TailNum"]


class DataProcessor(object):

    def drop_forbidden_and_excluded_variables(self, dataset):
        return dataset.drop(FORBIDDEN_VARS + EXCLUDED_VARS, axis=1)

    def drop_duplicated_data(self, dataset):
        return dataset.drop_duplicates()

    def remove_null_arr_delay(self, dataset):
        return dataset.dropna(subset=['ArrDelay'])

    def remove_cancelled_flights(self, dataset):
        no_cancelled_flights = dataset[dataset['Cancelled'] == 0]
        no_cancelled_flights.pop('Cancelled')
        return no_cancelled_flights

    def convert_date_fields(self, dataset):
        def convert_string_to_hour_and_minute(str_time):
            return (dataset[str_time] // 100).astype(int),  (dataset[str_time] % 100).astype(int)

        # check if it was already processed
        if 'DepTS' in dataset and 'CSRDepTS' in dataset:
            return dataset

        dataset['DepHour'], dataset['DepMinute'] = convert_string_to_hour_and_minute(dataset['DepTime'])
        dataset['CSRDepHour'], dataset['CSRDepMinute'] = convert_string_to_hour_and_minute(dataset['CRSDepTime'])

        dataset.pop('DepTime')
        dataset.pop('CRSDepTime')
        return dataset

    def run_all_data_processing(self, dataset):
        dataset = self.drop_forbidden_and_excluded_variables(dataset)
        dataset = self.remove_cancelled_flights(dataset)
        dataset = self.drop_duplicated_data(dataset)
        dataset = self.remove_null_arr_delay(dataset)
        dataset = self.convert_date_fields(dataset)

        print("[PROCESSING]: Finished the data processing...")
        print(dataset.columns)
        return dataset


data_processor = DataProcessor()