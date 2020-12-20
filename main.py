import os
from pyspark.sql import SparkSession

from load_data import DataLoader
from process_data import DataProcessor
from train_data import DataTrainer


LINEAR_REGRESSION = 1
DECISION_TREE = 2
RANDOM_FOREST = 3

ALGORITHM_OPTIONS = [LINEAR_REGRESSION, DECISION_TREE, RANDOM_FOREST]


class MachineLearningRunner(object):
    def __init__(self, algorithm):
        self.spark = SparkSession.builder.appName('Delay Classifier').master('local[*]').getOrCreate()
        self.dl = DataLoader(self.spark)
        self.dp = DataProcessor(self.spark)
        self.dt = DataTrainer(self.spark, self.dp)
        self.algorithm = algorithm

    def load_data(self, filepath):
        return self.dl.load_dataset(filepath)

    def process_data(self, input_dataset):
        return self.dp.run_all_data_processing(input_dataset)

    def train_data(self, processed_data):
        if self.algorithm == LINEAR_REGRESSION:
            return self.dt.linear_regression(processed_data)
        elif self.algorithm == DECISION_TREE:
            return self.dt.decision_tree(processed_data)
        return self.dt.random_forest(processed_data)

    def run(self, data_filepath):
        input_dataset = self.load_data(data_filepath)
        processed_data = self.process_data(input_dataset)
        saved_model = self.train_data(processed_data)
        print("Finished!")


class Launcher(object):
    def check_filepath(self, filepath):
        # TODO: Are there any exceptions to take care of?

        # CASE 1: File does not exist
        try:
            f = open(filepath)
        except IOError:
            return False, "No such file"
        else:
            f.close()

        # CASE 2: Not in format of csv
        if not filepath.lower().endswith('csv'):
            print(filepath.lower())
            return False, "Not in CSV"

        # CASE 3 : empty file
        if os.stat(filepath).st_size == 0:
            return False, "Empty File"
        return True, None

    def check_algorithm(self, algorithm):
        # TODO: Are there any exceptions to take care of?

        if not algorithm.isnumeric():
            return False, "algorithm choice should be a number"

        if int(algorithm) not in ALGORITHM_OPTIONS:
            return False, "algorithm should be {}, {} or {}".format(*ALGORITHM_OPTIONS)
        return True, None


launcher = Launcher()


if __name__ == '__main__':
    with open("spark-ml-ascii", "r") as f:
        print(f.read())

    # default values
    correct_syntax = False
    data_filepath = os.path.join(os.getcwd(), 'input_dataset', 'dataset.csv')
    algorithm = LINEAR_REGRESSION

    while not correct_syntax:
        # filepath for raw dataset
        data_filepath_input = input("Enter the filepath of a CSV dataset (by default, input_dataset/dataset.csv): ")

        if data_filepath_input:
            right_file_path, f_error_msg = launcher.check_filepath(data_filepath_input)
            if not right_file_path:
                print(f_error_msg + ", Enter again...")
                continue
            data_filepath = data_filepath_input

        # algorithm for machine learning
        algorithm_input = input("Enter the machine learning algorithm (1: Linear Regression, 2:Decision Tree, 3:Random Forest, "
                          "by default Linear Regression): ")

        if algorithm_input:
            algorithm_check, a_error_msg = launcher.check_algorithm(algorithm_input)
            if not algorithm_check:
                print(a_error_msg + ", Enter again...")
                continue
            algorithm = int(algorithm_input)
        correct_syntax = True

    ml_runner = MachineLearningRunner(algorithm)
    ml_runner.run(data_filepath)


