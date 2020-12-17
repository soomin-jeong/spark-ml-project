# Import all needed functions
import time
import sys
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col, floor

# SET UP OF ENVIRONMENT
# Create a Spark Session and check its version
spark = SparkSession.builder.appName('Delay Classifier').master('local[*]').getOrCreate()
print("Spark version", spark.version)

# Input variables provided by the user
file_path = "2007.csv.bz2"
reg_type = "linear_reg"
modelPath = reg_type + "3"

# Handle input data
try:
    airports = spark.read.csv(file_path, header=True)
except:
    print('Error occurred')
    sys.exit(1)

# Delete the forbidden columns
airports = airports.drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay",
                         "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")

# FILTER ON CANCELLED
airports = airports.where(airports.Cancelled.isNotNull())

# Drop irrelevant features
airports = airports.drop("CancellationCode", "Cancelled")

# Filter all the rows where ArrDelay is unknown
airports = airports.where(airports.ArrDelay.isNotNull())
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

# Transform strings to categories
indexer = StringIndexer(inputCols=['UniqueCarrier'], outputCols=['carrierIndex'])

# One hot encoding of categories
encoder = OneHotEncoder(inputCols=['ProcDepTime', 'ProcArrTime', 'carrierIndex'],
                        outputCols=['DepTimeVec', 'ArrTimeVec', 'carrierVec'])

# FEATURE SELECTION
input_cols = ['DepDelay', "TaxiOut", "carrierVec", 'DepTimeVec', 'ArrTimeVec']
assembler = VectorAssembler(inputCols=input_cols, outputCol='features')

# MACHINE LEARNING
# Split data
flights_train, flights_test = airports.randomSplit([0.75, 0.25], seed=123)
flights_train.show()

# Set up the regressor for choice of user and define the parameters for hyperparameter tuning
if reg_type == "random_forest":
    reg = RandomForestRegressor(featuresCol='features', labelCol='ArrDelay')

    params = ParamGridBuilder().addGrid(reg.numTrees, [4]) \
        .addGrid(reg.maxDepth, [6]) \
        .build()
elif reg_type == "dec_tree":
    reg = DecisionTreeRegressor(featuresCol='features', labelCol='ArrDelay', impurity='variance')

    params = ParamGridBuilder().addGrid(reg.maxDepth, [3]) \
        .addGrid(reg.maxBins, [15]) \
        .build()
else:
    reg = LinearRegression(featuresCol='features', labelCol='ArrDelay')

    params = ParamGridBuilder().addGrid(reg.regParam, [0, 0.5]) \
        .addGrid(reg.elasticNetParam, [0, 0.5, 1]) \
        .build()

# Construct a pipeline of preprocessing, feature engineering, selection and regressor
pipeline = Pipeline(stages=[indexer, encoder, assembler, reg])

# Error measure
rmse = RegressionEvaluator(labelCol='ArrDelay')

# Start the actual modelling
print("Start Model Fitting")
start = time.time()
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=rmse, numFolds=3)
cv_model = cv.fit(flights_train)
print("Time taken to develop model: " + str(time.time()-start) + 's.')

# Save the model
cv_model.write().overwrite().save(modelPath)

# Evaluation of the best model according to evaluator
print("RMSE training: " + str(rmse.evaluate(cv_model.bestModel.transform(flights_train))))
print('RMSE test: ' + str(rmse.evaluate(cv_model.bestModel.transform(flights_test))))

# Stop Spark
spark.stop()
