

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, VectorIndexer
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


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

    def random_forest(self, data):
        # Split the data into training and test sets (30% held out for testing)
        (trainingData, testData) = data.randomSplit([0.7, 0.3])

        input_cols = data.schema.names
        input_cols.remove('ArrDelay')

        stringIndexer = StringIndexer(inputCol='UniqueCarrier', outputCol='carrierIndex')
        input_cols.append('carrierIndex')
        input_cols.remove('UniqueCarrier')

        vectorAssember = VectorAssembler(inputCols=input_cols, outputCol="features")
        featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=31)

        # Train a RandomForest model.
        rf = RandomForestRegressor(featuresCol="indexedFeatures", labelCol="ArrDelay") #maxBins=8, numTrees=2)

        # Chain indexer and forest in a Pipeline
        pipeline = Pipeline(stages=[stringIndexer, vectorAssember, featureIndexer, rf])
        #
        # # Train model.  This also runs the indexer.
        # model = pipeline.fit(trainingData)
        #
        # # Make predictions.
        # predictions = model.transform(testData)
        #
        # # Select example rows to display.
        # predictions.select("prediction", "ArrDelay", "indexedFeatures").show(5)
        #
        # # Select (prediction, true label) and compute test error
        # evaluator = RegressionEvaluator(
        #     labelCol="ArrDelay", predictionCol="prediction", metricName="rmse")
        #
        # rmse = evaluator.evaluate(predictions)
        # print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
        #
        # rfModel = model.stages[1]
        # print(rfModel)  # summary only

        # Cross Validation
        # We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
        # This will allow us to jointly choose parameters for all Pipeline stages.
        # A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
        # We use a ParamGridBuilder to construct a grid of parameters to search over.
        # With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
        # this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
        paramGrid = ParamGridBuilder()\
            .addGrid(RandomForestRegressor(featuresCol="indexedFeatures", labelCol="ArrDelay").maxDepth, [3, 10, 20]) \
            .addGrid(RandomForestRegressor(featuresCol="indexedFeatures", labelCol="ArrDelay").maxBins, [20, 25]) \
            .build()

        crossval = CrossValidator(estimator=pipeline,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=RegressionEvaluator(labelCol="ArrDelay"),
                                  numFolds=3)

        # Run cross-validation, and choose the best set of parameters.
        cvModel = crossval.fit(trainingData)

        # Make predictions on test documents. cvModel uses the best model found (lrModel).
        prediction = cvModel.transform(testData)
        selected = prediction.select("prediction", "ArrDelay", "indexedFeatures")
        for row in selected.collect():
            print(row)

