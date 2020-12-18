import time

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession

from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, VectorIndexer


class DataTrainer(object):
    def __init__(self, spark, dp):
        self.spark = spark
        self.dp = dp

    def linear_regression(self, data):
        # train the data
        # print the RMSE
        print("[TRAIN] Linear Regression: printing the evaluation results...")

    # Arturo
    # Special processing data part:

    def preparePipelineForModel(self, inputIndexer, outputIndexer, inputEncoder,
                                outputEncoder, inputAssembler, outputAssembler, reg):
        indexer = self.dp.transformStringToCategories(inputIndexer, outputIndexer)
        encoder = self.dp.oneHotEncoder(inputEncoder, outputEncoder)

        # Feature selection:
        assembler = self.dp.vectorAssembler(inputAssembler, outputAssembler)

        # Construct a pipeline of preprocessing, feature engineering, selection and regressor
        pipeline = Pipeline(stages=[indexer, encoder, assembler, reg])
        return pipeline


    def decision_tree(self, data):
        # train the data
        # print the RMSE
        # Transform strings to categories

        # Input variables:
        modelPath = "linear_reg" + "3"

        # MACHINE LEARNING
        # Split data
        flights_train, flights_test = data.randomSplit([0.75, 0.25], seed=123)
        #flights_train.show()

        reg = DecisionTreeRegressor(featuresCol='features', labelCol='ArrDelay', impurity='variance')

        confParams = {}
        confParams.update({"maxDepth": [3, 10, 15]})
        confParams.update({"maxBins":[10, 15]})

        params = ParamGridBuilder().addGrid(reg.maxDepth, confParams.get("maxDepth")) \
                .addGrid(reg.maxBins, confParams.get("maxBins")) \
                .build()
        # Construct a pipeline of preprocessing, feature engineering, selection and regressor

        inputIndexer = ['UniqueCarrier']
        outputIndexer = ['carrierIndex']
        inputEncoder = ['ProcArrTime', 'carrierIndex']
        outputEncoder = ['DepTimeVec', 'ArrTimeVec', 'carrierVec']
        inputAssembler = ['DepDelay', "TaxiOut", "carrierVec", 'DepTimeVec', 'ArrTimeVec']
        outputAssembler = 'features'

        pipeline = self.preparePipelineForModel(inputIndexer, outputIndexer, inputEncoder,
                                                outputEncoder, inputAssembler, outputAssembler, reg)

        # Error measure:
        rmse = RegressionEvaluator(labelCol='ArrDelay')

        # Start the actual modelling:
        print("[TRAINING] Starting model Fitting")
        start = time.time()
        cv = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=rmse, numFolds=3)
        cv_model = cv.fit(flights_train)
        print("Time taken to develop model: " + str(time.time() - start()) + 's.')
        # Save the model
        cv_model.write().overwrite().save(modelPath)
        confParams.update({"modelPath": modelPath})

        # Evaluation of the best model according to evaluator
        print("RMSE training: " + str(rmse.evaluate(cv_model.bestModel.transform(flights_train))))
        print('RMSE test: ' + str(rmse.evaluate(cv_model.bestModel.transform(flights_test))))
        return confParams




    def trans_data(self, data):
        print("Transforming the data...")
        return data.rdd.map(lambda r: [Vectors.dense(r[1:8] + r[9:]), r[8]]).toDF(['features', 'label'])

    def random_forest(self, data):
        # SET UP OF ENVIRONMENT
        # Create a Spark Session
        #spark = SparkSession.builder.config("spark.driver.memory", "15g").appName('Delay Classifier').master('local[*]').getOrCreate()
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
        self.spark.stop()



