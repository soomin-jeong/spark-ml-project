import time

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator

from process_data import data_processor as dp
from load_data import data_loader as dl

from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col


class DataTrainer(object):
    def get_dummy(self, df, categoricalCols, continuousCols, labelCol):

        indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
                    for c in categoricalCols]

        # default setting: dropLast=True
        encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(),
                                  outputCol="{0}_encoded".format(indexer.getOutputCol()))
                    for indexer in indexers]

        assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders]
                                              + continuousCols, outputCol="features")

        pipeline = Pipeline(stages=indexers + encoders + [assembler])

        model = pipeline.fit(df)
        data = model.transform(df)

        data = data.withColumn('label', col(labelCol))

        return data.select('features', 'label')

    # def trans_data(self, data):
    #     return data.rdd.map(lambda r: [Vectors.dense(r[1:]), r[0]]).toDF(['features', 'label'])

    def random_forest(self):
        # SET UP OF ENVIRONMENT
        # Create a Spark Session
        spark = SparkSession.builder.config("spark.driver.memory", "15g").appName('Delay Classifier').master('local[*]').getOrCreate()

        # Load the data
        raw_dataset = dl.load_dataset()
        pd_data = dp.run_all_data_processing(raw_dataset)
        pd_data.pop('UniqueCarrier')
        pd_data.pop('DepTime')
        pd_data.pop('CRSDepTime')
        pd_data.pop('CRSArrTime')

        df_data = spark.createDataFrame(pd_data)

        categoricalCols = ['Month', 'DayofMonth', 'DayOfWeek']
        continuousCols = ['CRSElapsedTime', 'DepDelay', 'Distance', 'TaxiOut', 'DepHour', 'DepMinute', 'CSRDepHour', 'CSRDepMinute']
        labelCol = ['ArrDelay']

        transformed_data = self.get_dummy(df_data, categoricalCols, continuousCols, labelCol)
        transformed_data.show(5)

        # Split the data into training and test sets (30% held out for testing)
        (trainingData, testData) = transformed_data.randomSplit([0.7, 0.3])

        # Train a RandomForest model.
        rf = RandomForestRegressor()

        # Chain indexer and forest in a Pipeline
        pipeline = Pipeline(stages=[rf])

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

        spark.stop()


data_trainer = DataTrainer()
data_trainer.random_forest()