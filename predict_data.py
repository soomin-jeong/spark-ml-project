from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.tuning import CrossValidatorModel, ParamGridBuilder


class DataPredictor(object):
    def __init__(self, spark):
        self.spark = spark

    def predict_dt(self, dataset, confParams):

        dataset_train, dataset_test = dataset.randomSplit([0.75, 0.25], seed=123)
        cvModel = CrossValidatorModel.load(confParams.get("modelPath"))

        paramGrid = ParamGridBuilder().addGrid(DecisionTreeRegressor().maxDepth, confParams.get("maxDepth")) \
            .addGrid(DecisionTreeRegressor().maxBins, confParams.get("maxBins")) \
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
        pass


