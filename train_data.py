import time

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator

from process_data import data_processor as dp
from load_data import data_loader as dl


class DataTrainer(object):

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

        pd_data.pop('Origin')
        pd_data.pop('Dest')


        data = spark.createDataFrame(pd_data)

        data = data.withColumn('ArrDelay', data['ArrDelay'].cast(IntegerType())) \
            .withColumn('DepDelay', data['DepDelay'].cast(IntegerType())) \
            .withColumn('Distance', data['Distance'].cast(IntegerType())) \
            .withColumn('DayOfWeek', data['DayOfWeek'].cast(IntegerType())) \
            .withColumn('Month', data['Month'].cast(IntegerType())) \
            .withColumn('DepHour', data['DepHour'].cast(IntegerType())) \
            .withColumn('DepMinute', data['DepMinute'].cast(IntegerType())) \
            .withColumn('CSRDepHour', data['CSRDepHour'].cast(IntegerType())) \
            .withColumn('CSRDepMinute', data['CSRDepMinute'].cast(IntegerType()))

        stringIndexer = \
            StringIndexer(inputCols=['DayOfWeek', 'Month'],
                          outputCols=['DayOfWeek_idx', 'Month_idx'])

        # Split the data into training and test sets (30% held out for testing)
        (trainingData, testData) = data.randomSplit([0.7, 0.3])

        # Train a RandomForest model..
        rf = RandomForestRegressor()

        # Chain indexer and forest in a Pipeline
        pipeline = Pipeline(stages=[stringIndexer, rf])

        print("64")

        # Train model.  This also runs the indexer.
        model = pipeline.fit(trainingData)

        # Make predictions.
        predictions = model.transform(testData)

        # Select example rows to display.
        predictions.select("prediction", "ArrDelay").show(5)

        # Select (prediction, true label) and compute test error
        evaluator = RegressionEvaluator(
            labelCol="ArrDelay", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

        rfModel = model.stages[1]
        print(rfModel)  # summary only

        spark.stop()


data_trainer = DataTrainer()
data_trainer.random_forest()