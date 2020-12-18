import sys
import os
from pyspark.sql import SparkSession

from load_data import DataLoader
from process_data import DataProcessor
from train_data import DataTrainer
from predict_data import DataPredictor


LINEAR_REGRESSION = 1
DECEISION_TREE = 2
RANDOM_FOREST = 3

ALGORITHM_OPTIONS = [LINEAR_REGRESSION, DECEISION_TREE, RANDOM_FOREST]


class Launcher(object):

    def __init__(self):
        self.spark = SparkSession.builder.appName('Delay Classifier').master('local[*]').getOrCreate()
        self.dl = DataLoader(self.spark)
        self.dp = DataProcessor(self.spark)
        self.dt = DataTrainer(self.spark, self.dp)
        self.dpr = DataPredictor(self.spark)

    def load_data(self, filepath):
        return self.dl.load_dataset(filepath)

    def process_data(self, input_dataset):
        return self.dp.run_all_data_processing(input_dataset)

    def train_data(self, processed_data, model):
        if model == LINEAR_REGRESSION:
            # linear regression
            self.dt.linear_regression(processed_data)
        elif model == "dt":
            # decision tree
            confParams = self.dt.decision_tree(processed_data)
        else:
            confParams = self.dt.random_forest(processed_data)
        return confParams

    def predict(self, processed_data, confParams, model):
        if model == "lr":
            self.dpr.predict_lr(processed_data, confParams)
        elif model == "dt":
            # decision tree:
            self.dpr.predict_dt(processed_data, confParams)
        else:
            self.dpr.predict_rf(processed_data, confParams)

    def run(self, data_filepath, algorithm):
        input_dataset = self.load_data(data_filepath)
        processed_data = self.process_data(input_dataset)
        confParams = self.train_data(processed_data, algorithm)
        self.predict(processed_data, confParams, algorithm)
        print("Finished!")

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

        if algorithm not in ALGORITHM_OPTIONS:
            return False, "algorithm should be {}, {} or {}".format(*ALGORITHM_OPTIONS)
        return True, None

    def launch_app(self, data_filepath, algorithm):
        runner = Launcher()
        runner.run(data_filepath, algorithm)


launcher = Launcher()


if __name__ == '__main__':
    with open("spark-ml-ascii", "r") as f:
        print(f.read())

    # default values
    correct_syntax = False
    data_filepath = os.path.join(os.getcwd(), 'input_dataset', 'dataset.csv')
    algorithm = LINEAR_REGRESSION

    while not correct_syntax:
        data_filepath_input = input("Enter the filepath of a CSV dataset (by default, input_dataset/dataset.csv): ")
        algorithm_input = input("Enter the machine learning algorithm (1: Linear Regression, 2:Decision Tree, 3:Random Forest, "
                          "by default linear Regression): ")

        if data_filepath_input:
            data_filepath = data_filepath_input
        if algorithm_input:
            algorithm = int(algorithm_input)

        right_file_path, f_error_msg = launcher.check_filepath(data_filepath)
        if not right_file_path:
            print(f_error_msg + ", Enter again...")
            continue

        algorithm_check, a_error_msg = launcher.check_algorithm(algorithm)
        if not algorithm_check:
            print(a_error_msg + ", Enter again...")
            continue
        correct_syntax = True

    launcher.launch_app(data_filepath, algorithm)


