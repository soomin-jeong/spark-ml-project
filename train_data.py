
import time

from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import CrossValidatorModel, ParamGridBuilder

MAX_DEPTH_OPTIONS = [3, 10, 15]
MAX_BINS_OPTIONS = [10, 15]
TARGET_VARIABLE = 'ArrDelay'


class DataTrainer(object):

    def __init__(self, spark, dp):
        self.spark = spark
        self.dp = dp

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

    def linear_regression(self, data):
        print("[TRAIN] Linear Regression: printing the evaluation results...")

        # TODO : [TOM] Linear Regression Model Path
        return "dummy model path"

    def abstract_decision_tree(self, train_data, regressor, model_path):
        params = ParamGridBuilder().addGrid(regressor.maxDepth, MAX_DEPTH_OPTIONS) \
                .addGrid(regressor.maxBins, MAX_BINS_OPTIONS) \
                .build()

        # Construct a pipeline of preprocessing, feature engineering, selection and regressor
        pipeline = self.build_pipeline(self.dp.ordinal_category_vars, self.dp.nominal_category_vars,
                                       self.dp.numerical_vars, regressor)

        # Error measure:
        rmse = RegressionEvaluator(labelCol=TARGET_VARIABLE)

        # Start the actual modelling:
        print("[TRAINING] Starting model Fitting")
        start = time.time()
        cv = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=rmse, numFolds=3)
        cv_model = cv.fit(train_data)

        print("Time taken to develop model: " + str(time.time() - start) + 's.')
        # Save the model
        cv_model.write().overwrite().save(model_path)

        # Evaluation of the best model according to evaluator
        print("RMSE training: " + str(rmse.evaluate(cv_model.bestModel.transform(train_data))))
        return model_path

    def cross_validate(self, test_data, regressor, saved_model):
        cvModel = CrossValidatorModel.load(saved_model)

        paramGrid = ParamGridBuilder().addGrid(regressor.maxDepth, MAX_DEPTH_OPTIONS) \
            .addGrid(regressor.maxBins, MAX_BINS_OPTIONS) \
            .build()

        # Evaluation of the best model according to evaluator
        for perf, params in zip(cvModel.avgMetrics, paramGrid):
            for param in params:
                print(param.name, params[param], end=' ')
            print(' achieved a performance of ', perf, 'RMSE')

        rmse = RegressionEvaluator(labelCol=TARGET_VARIABLE)
        print('Performance Best Model')
        print("RMSE training: " + str(min(cvModel.avgMetrics)))
        print('RMSE test: ' + str(rmse.evaluate(cvModel.bestModel.transform(test_data))))
        self.spark.stop()

    def decision_tree(self, data):
        # TODO: Explain why we dropped UniqueCarrier on the report
        data = data.drop("UniqueCarrier")
        regressor = DecisionTreeRegressor(featuresCol='features', labelCol=TARGET_VARIABLE, impurity='variance')
        model_path = "decision_tree" + "3"

        train_data, test_data = self.get_training_and_test_data(data)
        self.abstract_decision_tree(train_data, regressor, model_path)
        self.cross_validate(test_data, regressor, model_path)

    def random_forest(self, data):
        # TODO: Explain why we dropped UniqueCarrier on the report
        data.drop("UniqueCarrier")
        regressor = RandomForestRegressor(featuresCol='features', labelCol=TARGET_VARIABLE)
        model_path = "random_forest" + "3"
        
        train_data, test_data = self.get_training_and_test_data(data)
        self.abstract_decision_tree(train_data, regressor, model_path)
        self.cross_validate(test_data, regressor, model_path)

