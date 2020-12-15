from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

from process_data import data_processor as dp


class DataTrainer(object):

    def ramdom_forest(self, dataset):
        return


data_trainer = DataTrainer()