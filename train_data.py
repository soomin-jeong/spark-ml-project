
import time

from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, VectorIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


class DataTrainer(object):
    def __init__(self, spark, dp):
        self.spark = spark
        self.dp = dp

    def linear_regression(self, data):
        # train the data
        # print the RMSE
        print("[TRAIN] Linear Regression: printing the evaluation results...")

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
        print("Time taken to develop model: " + str(time.time() - start) + 's.')
        # Save the model
        cv_model.write().overwrite().save(modelPath)
        confParams.update({"modelPath": modelPath})

        # Evaluation of the best model according to evaluator
        print("RMSE training: " + str(rmse.evaluate(cv_model.bestModel.transform(flights_train))))
        print('RMSE test: ' + str(rmse.evaluate(cv_model.bestModel.transform(flights_test))))
        return confParams


    def random_forest(self, data):
        featureIndexer = \
            VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=31).fit(data)

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

