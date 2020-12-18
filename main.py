
from pyspark.sql import SparkSession

from load_data import DataLoader
from process_data import DataProcessor
from train_data import DataTrainer
from predict_data import DataPredictor


class ArrivalDelayMachineLearningRunner(object):

    def __init__(self):
        self.spark = SparkSession.builder.appName('Delay Classifier').master('local[*]').getOrCreate()
        self.dl = DataLoader(self.spark)
        self.dp = DataProcessor(self.spark)
        self.dt = DataTrainer(self.spark)
        self.dpr = DataPredictor(self.spark)

    def load_data(self):
        self.dl.locate_input_data()
        return self.dl.load_dataset()

    def process_data(self, input_dataset):
        return self.dp.run_all_data_processing(input_dataset)

    def train_data(self, processed_data):
        # linear regression
        self.dt.linear_regression(processed_data)

        # decision tree
        self.dt.decision_tree(processed_data)

        # random forest
        self.dt.random_forest(processed_data)
        return

    def predict(self, processed_data):
        self.dpr.predict_dummy_function(processed_data)

    def run(self):
        input_dataset = self.load_data()
        processed_data = self.process_data(input_dataset)
        self.train_data(processed_data)
        # self.predict(processed_data)
        print("Finished!")


runner = ArrivalDelayMachineLearningRunner()
runner.run()
