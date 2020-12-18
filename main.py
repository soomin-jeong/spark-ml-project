
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
        self.dt = DataTrainer(self.spark, self.dp)
        self.dpr = DataPredictor(self.spark)

    def load_data(self, dataPath):
        self.dl.setFilePath(dataPath)
        self.dl.locate_input_data()
        return self.dl.load_dataset()

    def process_data(self, input_dataset):
        return self.dp.run_all_data_processing(input_dataset)

    def train_data(self, processed_data, model):
        if model == "lr":
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

    def run(self, appConf):
        input_dataset = self.load_data(appConf.get("data"))
        processed_data = self.process_data(input_dataset)
        confParams = self.train_data(processed_data, appConf.get("model"))
        self.predict(processed_data, confParams, appConf.get("model"))
        print("Finished!")

