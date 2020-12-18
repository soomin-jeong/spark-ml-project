import time

from pyspark.sql import SparkSession

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, VectorIndexer


class DataTrainer(object):
    def __init__(self, spark):
        self.spark = spark

    def linear_regression(self, data):
        # train the data
        # print the RMSE
        print("[TRAIN] Linear Regression: printing the evaluation results...")

    def decision_tree(self, data):
        # train the data
        # print the RMSE
        print("[TRAIN] Decision Tree: printing the evaluation results...")

    def trans_data(self, data):
        print("Transforming the data...")
        return data.rdd.map(lambda r: [Vectors.dense(r[1:8] + r[9:]), r[8]]).toDF(['features', 'label'])

    def random_forest(self, data):
        transformed_data = self.trans_data(data)

        featureIndexer = \
            VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=31).fit(transformed_data)

        # Split the data into training and test sets (30% held out for testing)
        (trainingData, testData) = transformed_data.randomSplit([0.7, 0.3])

        # Train a RandomForest model.
        rf = RandomForestRegressor(featuresCol="indexedFeatures", maxBins=8, numTrees=2)

        # Chain indexer and forest in a Pipeline
        pipeline = Pipeline(stages=[featureIndexer, rf])

        # Train model.  This also runs the indexer.
        model = pipeline.fit(trainingData)

        # Make predictions.
        predictions = model.transform(testData)

        # Select example rows to display.
        predictions.select("prediction", "label", "features").show(5)

        # Select (prediction, true label) and compute test error
        evaluator = RegressionEvaluator(
            labelCol="label", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

        rfModel = model.stages[1]
        print(rfModel)  # summary only



