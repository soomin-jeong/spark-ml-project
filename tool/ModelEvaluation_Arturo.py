from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import IntegerType
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.sql.functions import col, floor
import sys

# SET UP OF ENVIRONMENT
# Create a Spark Session
spark = SparkSession.builder.appName('Delay Classifier').master('local[*]').getOrCreate()

# Print the version
print("Spark version", spark.version)

file_path = "2007.csv.bz2"

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
# FEATURE ENGINGEERING
airports = airports.withColumn('CRSDepTime', airports['CRSDepTime'].cast(IntegerType())) \
                    .withColumn('ArrDelay', airports['ArrDelay'].cast(IntegerType())) \
                    .withColumn('DepDelay', airports['DepDelay'].cast(IntegerType())) \
                    .withColumn('Distance', airports['Distance'].cast(IntegerType())) \
                    .withColumn('DayOfWeek', airports['DayOfWeek'].cast(IntegerType())) \
                    .withColumn('Month', airports['Month'].cast(IntegerType())) \
                    .withColumn('TaxiOut', airports['TaxiOut'].cast(IntegerType())) \
                    .withColumn('DepTime', airports['DepTime'].cast(IntegerType())) \
                    .withColumn('CRSArrTime', airports['CRSArrTime'].cast(IntegerType()))

airports = airports.where(airports.ArrDelay.isNotNull())
airports = airports.withColumn('ProcDepTime', floor(col('DepTime')/100)) \
                    .withColumn('ProcArrTime', floor(col('CRSArrTime')/100))

# FEATURE SELECTION
input_cols = ['DepDelay', "TaxiOut", "carrierIndex", 'DepTime', 'CRSArrTime']
# Split data (Change values according to the values in tool)
flights_train, flights_test = airports.randomSplit([0.75, 0.25], seed=123)

mPath = 'dec_tree3'
cvModel = CrossValidatorModel.load(mPath)

# This paramGrid should be copied from the tool
paramGrid = ParamGridBuilder().addGrid(DecisionTreeRegressor().maxDepth, [3, 10, 20]) \
        .addGrid(DecisionTreeRegressor().maxBins, [20, 25]) \
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