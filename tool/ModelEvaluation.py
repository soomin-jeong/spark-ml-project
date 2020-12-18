from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import IntegerType
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidatorModel
import sys


# SET UP OF ENVIRONMENT
# Create a Spark Session
spark = SparkSession.builder.appName('Delay Classifier').master('local[*]').getOrCreate()

# Print the version
print("Spark version", spark.version)

file_path = "1990.csv.bz2"

# Handle input data
try:
    if type(file_path) == str:
        airports = spark.read.csv(file_path, header=True)
    else:
        for file in file_path:
            airports = spark.read.csv(file, header=True)
except:
    print('Error occurred')
    sys.exit(1)

# Delete the forbidden columns
airports = airports.drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay",
                         "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")
print("Data loaded")
airports = airports.withColumn('CRSDepTime', airports['CRSDepTime'].cast(IntegerType())) \
                    .withColumn('ArrDelay', airports['ArrDelay'].cast(IntegerType())) \
                    .withColumn('DepDelay', airports['DepDelay'].cast(IntegerType())) \
                    .withColumn('Distance', airports['Distance'].cast(IntegerType())) \
                    .withColumn('DayOfWeek', airports['DayOfWeek'].cast(IntegerType())) \
                    .withColumn('Month', airports['Month'].cast(IntegerType()))

# Filter all the rows where ArrDelay is unknown
airports = airports.where(airports.ArrDelay.isNotNull())

# Split data (Change values according to the values in tool)
flights_train, flights_test = airports.randomSplit([0.05, 0.8], seed=123)#[0.9, 0.1], seed=123)

mPath = 'lr_model'
cvModel = CrossValidatorModel.load(mPath)

# This paramGrid should be copied from the tool
paramGrid = ParamGridBuilder().addGrid(LinearRegression.regParam, [0.01]) \
                        .addGrid(LinearRegression.elasticNetParam, [0.5])\
                        .build()

# Evaluation of the best model according to evaluator
for perf, params in zip(cvModel.avgMetrics, paramGrid):
    for param in params:
        print(param.name, params[param], end=' ')
    print(' achieved a performance of ', perf, 'RMSE')

rmse = RegressionEvaluator(labelCol='ArrDelay')
print('Performance Best Model')
print("RMSE training: " + str(min(cvModel.avgMetrics)))
print('RMSE test: ' + str(rmse.evaluate(cvModel.bestModel.transform(flights_test))))