
import time, sys

from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel

MAX_DEPTH_OPTIONS = [5, 10]
MAX_BINS_OPTIONS = [10, 15]
REG_PARAM_OPTIONS = [0, 0.5]
ELASTICNET_PARAM_OPTIONS = [0, 0.5, 1]
RF_MAX_DEPTH_OPTIONS = [15]
RF_MAX_BINS_OPTIONS = [10, 15]

TARGET_VARIABLE = 'ArrDelay'
INPUT_LINREG = ["DepDelay", 'TaxiOut', "UniqueCarrier", "ProcArrTime", "ProcDepTime"]
INPUT_TREES = ['DepDelay', "TaxiOut", 'DepTime', 'CRSArrTime']

class DataTrainer(object):

    def __init__(self, spark, dp):
        self.spark = spark
        self.dp = dp
        self.evaluator = RegressionEvaluator(labelCol=TARGET_VARIABLE)

    def build_pipeline(self, reg):
        if reg == LinearRegression():
            print("You are running linear regression")
            input_string = ["UniqueCarrier"]
            input_onehot = ['ProcDepTime', 'ProcArrTime', 'carrier_idx']
            input_cols = INPUT_LINREG
        else:
            input_string = []
            input_onehot = []
            input_cols = INPUT_TREES

        indexer, _ = self.dp.string_indexer(input_string)
        encoder, _ = self.dp.string_indexer(input_onehot)
        assembler = self.dp.vector_assembler(input_cols)
        pipeline = Pipeline(stages=[indexer, encoder, assembler, reg])
        return pipeline

    def learn_from_training_data(self, data, regressor, model_path, param_grid):
        train_data, test_data = data.randomSplit([0.75, 0.25], seed=123)

        # Construct a pipeline of preprocessing, feature engineering, selection and regressor
        pipeline = self.build_pipeline(regressor)

        # Start the actual modelling:
        print("[TRAINING] Starting model Fitting")
        start = time.time()
        cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=self.evaluator, numFolds=3)
        cv_model = cv.fit(train_data)

        print("[TRAINING] Time taken to develop model: " + str(time.time() - start) + 's.')

        # Save the model
        cv_model.write().overwrite().save(model_path)
        print("[MODEL] Cross Validation Model is saved at", model_path)
        print("[EVALUATION] RMSE best model in Cross Validation: " + str(min(cv_model.avgMetrics)))
        print("[EVALUATION] RMSE best model on test: " + str(self.evaluator.evaluate(cv_model.bestModel.transform(test_data))))

        # Hyperparameter evaluation
        self.evaluate_hyperparameters(test_data,  cv_model, param_grid)

    def evaluate_hyperparameters(self, test_data, cvModel, param_grid):
        # Evaluation of the best model according to evaluator
        for perf, params in zip(cvModel.avgMetrics, param_grid):
            for param in params:
                print(param.name, params[param])
            print(' achieved a performance of ', perf, 'RMSE')

    def linear_regression(self, data):
        model_path = 'linear_regression'
        regressor = LinearRegression(featuresCol='features', labelCol='ArrDelay')

        # Define the parameter grid for hyperparameter tuning
        param_grid = ParamGridBuilder().addGrid(regressor.regParam, REG_PARAM_OPTIONS) \
            .addGrid(regressor.elasticNetParam, ELASTICNET_PARAM_OPTIONS) \
            .build()

        # learning on training data
        self.learn_from_training_data(data, regressor, model_path, param_grid)

    def tree_builder(self, data, algorithm):
        print(algorithm)
        if algorithm == 2:
            regressor = DecisionTreeRegressor(featuresCol='features', labelCol=TARGET_VARIABLE, impurity='variance')
            model_path = "decision_tree"
        elif algorithm == 3:
            regressor = RandomForestRegressor(featuresCol='features', labelCol=TARGET_VARIABLE, numTrees=4)
            model_path = "random_forest"
            print("[REG] number of trees: ", regressor.getNumTrees())

        param_grid = ParamGridBuilder().addGrid(regressor.maxDepth, MAX_DEPTH_OPTIONS) \
            .addGrid(regressor.maxBins, MAX_BINS_OPTIONS) \
            .build()

        # learning on training data
        self.learn_from_training_data(data, regressor, model_path, param_grid)
