import os, sys
from pyspark.sql import SparkSession

from utilities.process_data import DataProcessor
from utilities.train_data import DataTrainer


LINEAR_REGRESSION = 1
DECISION_TREE = 2
RANDOM_FOREST = 3

ALGORITHM_OPTIONS = [LINEAR_REGRESSION, DECISION_TREE, RANDOM_FOREST]


class MachineLearningRunner(object):
    def __init__(self, algorithm, spark):
        self.spark = spark
        self.dp = DataProcessor(self.spark)
        self.dt = DataTrainer(self.spark, self.dp)
        self.algorithm = algorithm

    def load_data(self, filepath):
        print("[LOAD] Loading the dataset...")
        self.dataset = self.spark.read.csv(filepath, header=True)
        print("[LOAD] Successful")
        return self.dataset

    def process_data(self, input_dataset):
        print("[PROCESSING] Start executing...")
        self.dataset = self.dp.run_all_data_processing(input_dataset)
        print("[PROCESSING] Done processing the data")
        return self.dataset

    def train_data(self, processed_data):
        print(self.algorithm)
        if self.algorithm == LINEAR_REGRESSION:
            return self.dt.linear_regression(processed_data)
        else:
            return self.dt.tree_builder(processed_data, self.algorithm)

    def run(self, data_filepath):
        input_dataset = self.load_data(data_filepath)
        processed_data = self.process_data(input_dataset)
        saved_model = self.train_data(processed_data)
        print("[APPLICATION] Finished!")


class Launcher(object):
    def check_filepath(self, filepath):

        # CASE 1: File does not exist
        try:
            f = open(filepath)
        except IOError:
            return False, "No such file"
        else:
            f.close()

        # CASE 2: Not in format of csv
        if not filepath.lower().endswith('csv') and not filepath.lower().endswith('csv.bz2'):
            print(filepath.lower())
            return False, "[ERROR] Not in CSV or CSV.BZ2 format"

        # CASE 3 : empty file
        if os.stat(filepath).st_size == 0:
            return False, "[ERROR] Empty File"
        return True, None

    def check_algorithm(self, algorithm):
        try:
            int(algorithm)
        except:
            print("[ERROR] Algorithm is not an integer")
            sys.exit(1)
        if int(algorithm) not in ALGORITHM_OPTIONS:
            return False, "[ERROR] algorithm should be {}, {} or {}".format(*ALGORITHM_OPTIONS)
        return True, None


if __name__ == '__main__':

    # Use the input variables given by user
    arguments = sys.argv
    print(arguments)
    path = "/".join(arguments[0].split("/")[:-1])
    with open(os.path.join(path + "/spark-ml-ascii"), "r") as f:
        print(f.read())

    launcher = Launcher()

    
    data_filepath = os.path.join(path, 'input_dataset', arguments[1])
    print(data_filepath)
    algorithm = arguments[2]

    # Check the input variables given by user
    bool_file, error_msg = launcher.check_filepath(data_filepath)
    if not bool_file:
        print(error_msg)
        sys.exit(1)
    
    bool_algo, error_msg = launcher.check_algorithm(algorithm)
    if not bool_algo:
        print(error_msg)
        sys.exit(1)

    algorithm = int(algorithm)

    # Run the spark application
    spark = SparkSession.builder.appName('Delay Classifier').master('local[*]').getOrCreate()
    ml_runner = MachineLearningRunner(algorithm, spark)
    ml_runner.run(data_filepath)
    spark.stop()



