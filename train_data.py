
import time

from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, VectorIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

MAX_DEPTH_OPTIONS = [3, 10, 15]
MAX_BINS_OPTIONS = [10, 15]


class DataTrainer(object):

    def __init__(self, spark, dp):
        self.spark = spark
        self.dp = dp

    def get_training_and_test_data(self, data):
        return data.randomSplit([0.75, 0.25], seed=123)

    def build_pipeline(self, ordial_categories, nominal_categories, numerical_vars, reg):
        s_indexer, s_output_cols = self.dp.string_indexer(ordial_categories)
        h_encoder, h_output_cols = self.dp.one_hot_encoder(nominal_categories)
        assembler_inputs = numerical_vars + s_output_cols + h_output_cols

        # Feature selection: returns an aseembler with "features"
        assembler = self.dp.vector_assembler(assembler_inputs)

        # Construct a pipeline of preprocessing, feature engineering, selection and regressor
        pipeline = Pipeline(stages=[s_indexer, h_encoder, assembler, reg])
        return pipeline

    def linear_regression(self, data):
        print("[TRAIN] Linear Regression: printing the evaluation results...")

        # TODO : [TOM] Linear Regression Model Path
        return "dummy model path"

    def abstract_decision_tree(self, data, regressor, model_path):

        # Split data
        train_data, test_data = self.get_training_and_test_data(data)
        params = ParamGridBuilder().addGrid(regressor.maxDepth, MAX_DEPTH_OPTIONS) \
                .addGrid(regressor.maxBins, MAX_BINS_OPTIONS) \
                .build()

        # Construct a pipeline of preprocessing, feature engineering, selection and regressor
        pipeline = self.build_pipeline(self.dp.ordinal_category_vars, self.dp.nominal_category_vars,
                                       self.dp.numerical_vars, regressor)

        # Error measure:
        rmse = RegressionEvaluator(labelCol='ArrDelay')

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
        print('RMSE test: ' + str(rmse.evaluate(cv_model.bestModel.transform(test_data))))
        return model_path

    def decision_tree(self, data):
        regressor = DecisionTreeRegressor(featuresCol='features', labelCol='ArrDelay', impurity='variance')
        model_path = "decision_tree" + "3"
        self.abstract_decision_tree(data, regressor, model_path)

    def random_forest(self, data):
        regressor = RandomForestRegressor(featuresCol='features', labelCol='ArrDelay')
        model_path = "random_forest" + "3"
        self.abstract_decision_tree(data, regressor, model_path)

