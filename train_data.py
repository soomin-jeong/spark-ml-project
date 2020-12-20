
import time

from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel

MAX_DEPTH_OPTIONS = [3, 10, 15]
MAX_BINS_OPTIONS = [10, 15]
REG_PARAM_OPTIONS = [0, 0.5]
ELASTICNET_PARAM_OPTIONS = [0, 0.5, 1]

TARGET_VARIABLE = 'ArrDelay'


class DataTrainer(object):

    def __init__(self, spark, dp):
        self.spark = spark
        self.dp = dp
        self.evaluator = RegressionEvaluator(labelCol=TARGET_VARIABLE)

    def get_training_and_test_data(self, data):
        return data.randomSplit([0.75, 0.25], seed=123)

    def build_pipeline(self, ordinal_categories, nominal_categories, numerical_vars, reg):
        # String index on ordinal categories
        s_indexer, s_output_cols = self.dp.string_indexer(ordinal_categories)

        # One hot-encoded indexer on nominal categories (Preproecssing with String indexer)
        s_indexer_nom, s_output_cols_nom = self.dp.string_indexer(nominal_categories)
        h_encoder, h_output_cols = self.dp.one_hot_encoder(s_output_cols_nom)

        # Assemble all the columns into one vector called 'feature'
        assembler_inputs = numerical_vars + s_output_cols + h_output_cols
        assembler = self.dp.vector_assembler(assembler_inputs)

        # Construct a pipeline of preprocessing, feature engineering, selection and regressor
        pipeline = Pipeline(stages=[s_indexer, s_indexer_nom, h_encoder, assembler, reg])
        return pipeline

    def learn_from_training_data(self, train_data, regressor, model_path, param_grid):
        # Construct a pipeline of preprocessing, feature engineering, selection and regressor
        pipeline = self.build_pipeline(self.dp.ordinal_category_vars, self.dp.nominal_category_vars,
                                       self.dp.numerical_vars, regressor)

        # Start the actual modelling:
        print("[TRAINING] Starting model Fitting")
        start = time.time()
        cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=self.evaluator, numFolds=3)
        cv_model = cv.fit(train_data)

        print("Time taken to develop model: " + str(time.time() - start) + 's.')

        # Save the model
        cv_model.write().overwrite().save(model_path)
        print("RMSE training: " + str(self.evaluator.evaluate(cv_model.bestModel.transform(train_data))))

    def drop_unique_carrier_due_to_memory_overload(self, data):
        data = data.drop("UniqueCarrier")
        self.dp.nominal_category_vars.remove("UniqueCarrier")
        return data

    def cross_validate(self, test_data, saved_model, param_grid):
        cvModel = CrossValidatorModel.load(saved_model)

        # Evaluation of the best model according to evaluator
        for perf, params in zip(cvModel.avgMetrics, param_grid):
            for param in params:
                print(param.name, params[param], end=' ')
            print(' achieved a performance of ', perf, 'RMSE')

        print('Performance Best Model')
        print("RMSE training: " + str(min(cvModel.avgMetrics)))
        print('RMSE test: ' + str(self.evaluator.evaluate(cvModel.bestModel.transform(test_data))))
        self.spark.stop()

    def linear_regression(self, data):

        regressor = LinearRegression(featuresCol='features', labelCol='ArrDelay')
        model_path = 'linear_regression'

        param_grid = ParamGridBuilder().addGrid(regressor.regParam, REG_PARAM_OPTIONS) \
            .addGrid(regressor.elasticNetParam, ELASTICNET_PARAM_OPTIONS) \
            .build()

        train_data, test_data = self.get_training_and_test_data(data)

        # learning on training data
        self.learn_from_training_data(train_data, regressor, model_path, param_grid)

        # evavluate on test data
        self.cross_validate(test_data, model_path, param_grid)

    def decision_tree(self, data):
        # TODO: Explain why we dropped UniqueCarrier on the report
        data = self.drop_unique_carrier_due_to_memory_overload(data)

        regressor = DecisionTreeRegressor(featuresCol='features', labelCol=TARGET_VARIABLE, impurity='variance')
        model_path = "decision_tree"

        param_grid = ParamGridBuilder().addGrid(regressor.maxDepth, MAX_DEPTH_OPTIONS) \
            .addGrid(regressor.maxBins, MAX_BINS_OPTIONS) \
            .build()

        train_data, test_data = self.get_training_and_test_data(data)

        # learning on training data
        self.learn_from_training_data(train_data, regressor, model_path, param_grid)

        # evavluate on test data
        self.cross_validate(test_data, model_path, param_grid)

    def random_forest(self, data):
        # TODO: Explain why we dropped UniqueCarrier on the report
        data = self.drop_unique_carrier_due_to_memory_overload(data)

        regressor = RandomForestRegressor(featuresCol='features', labelCol=TARGET_VARIABLE)
        model_path = "random_forest"

        param_grid = ParamGridBuilder().addGrid(regressor.maxDepth, MAX_DEPTH_OPTIONS) \
            .addGrid(regressor.maxBins, MAX_BINS_OPTIONS) \
            .build()

        train_data, test_data = self.get_training_and_test_data(data)

        # learning on training data
        self.learn_from_training_data(train_data, regressor, model_path, param_grid)

        # evavluate on test data
        self.cross_validate(test_data, model_path, param_grid)


