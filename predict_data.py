from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.tuning import CrossValidatorModel, ParamGridBuilder
from train_data import MAX_DEPTH_OPTIONS, MAX_BINS_OPTIONS


class DataPredictor(object):

    # TODO : Are we going to use the predictor?
    def __init__(self, spark):
        self.spark = spark

    def predict_dt(self, dataset, saved_model):

        dataset_train, dataset_test = dataset.randomSplit([0.75, 0.25], seed=123)
        cvModel = CrossValidatorModel.load(saved_model)

        paramGrid = ParamGridBuilder().addGrid(DecisionTreeRegressor().maxDepth, MAX_DEPTH_OPTIONS) \
            .addGrid(DecisionTreeRegressor().maxBins, MAX_BINS_OPTIONS) \
            .build()

        # Evaluation of the best model according to evaluator
        for perf, params in zip(cvModel.avgMetrics, paramGrid):
            for param in params:
                print(param.name, params[param], end=' ')
            print(' achieved a performance of ', perf, 'RMSE')

        rmse = RegressionEvaluator(labelCol='ArrDelay')
        print('Performance Best Model')
        print("RMSE training: " + str(min(cvModel.avgMetrics)))
        print('RMSE test: ' + str(rmse.evaluate(cvModel.bestModel.transform(dataset_test))))
        self.spark.stop()

    def predict_lr(self, dataset, confParams):
        pass

    def predict_rf(self, dataset, confParams):
        dataset_train, dataset_test = dataset.randomSplit([0.75, 0.25], seed=123)
        cvModel = CrossValidatorModel.load(saved_model)

        paramGrid = ParamGridBuilder().addGrid(DecisionTreeRegressor().maxDepth, MAX_DEPTH_OPTIONS) \
            .addGrid(DecisionTreeRegressor().maxBins, MAX_BINS_OPTIONS) \
            .build()

        # Evaluation of the best model according to evaluator
        for perf, params in zip(cvModel.avgMetrics, paramGrid):
            for param in params:
                print(param.name, params[param], end=' ')
            print(' achieved a performance of ', perf, 'RMSE')

        rmse = RegressionEvaluator(labelCol='ArrDelay')
        print('Performance Best Model')
        print("RMSE training: " + str(min(cvModel.avgMetrics)))
        print('RMSE test: ' + str(rmse.evaluate(cvModel.bestModel.transform(dataset_test))))
        self.spark.stop()


