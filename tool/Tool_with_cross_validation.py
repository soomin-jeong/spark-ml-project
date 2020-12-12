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

# SET UP OF ENVIRONMENT
# Create a Spark Session and check its version
spark = SparkSession.builder.appName('Delay Classifier').master('local[*]').getOrCreate()
print("Spark version", spark.version)

# Input variables provided by the user
file_path = "1990.csv.bz2"
reg_type = "dec_tree"
modelPath = reg_type

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
airports = airports.drop("CancellationCode")

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
                    #.withColumn('DepTime', airports['DepTime'].cast(IntegerType())) \

# Transform strings to categories
indexer = StringIndexer(inputCols=['UniqueCarrier'], outputCols=['carrierIndex'])

# One hot encoding of categories
encoder = OneHotEncoder(inputCols=['DayOfWeek', 'Month', 'carrierIndex'],
                        outputCols=['DayOfWeekVec', 'MonthVec', 'carrierVec'])

# FEATURE SELECTION
assembler = VectorAssembler(inputCols=[
    'DepDelay', 'DayOfWeekVec', 'MonthVec', 'carrierVec'
], outputCol='features')

# MACHINE LEARNING
# Split data
flights_train, flights_test = airports.randomSplit([0.05, 0.8], seed=123)#[0.9, 0.1], seed=123)
flights_train.show()

# Set up the regressor for choice of user and define the parameters for hyperparameter tuning
if reg_type == "random_forest":
    reg = RandomForestRegressor(featuresCol='features', labelCol='ArrDelay')

    params = ParamGridBuilder().addGrid(reg.numTrees, [3]) \
        .addGrid(reg.maxDepth, [5]) \
        .build()
elif reg_type == "dec_tree":
    reg = DecisionTreeRegressor(featuresCol='features', labelCol='ArrDelay', impurity='variance')

    params = ParamGridBuilder().addGrid(reg.maxDepth, [3]) \
        .addGrid(reg.maxBins, [15]) \
        .build()
else:
    reg = LinearRegression(featuresCol='features', labelCol='ArrDelay')

    params = ParamGridBuilder().addGrid(reg.regParam, [0.01]) \
        .addGrid(reg.elasticNetParam, [0.5]) \
        .build()

# Construct a pipeline of preprocessing, feature engineering, selection and regressor
pipeline = Pipeline(stages=[indexer, encoder, assembler, reg])

# Error measure
rmse = RegressionEvaluator(labelCol='ArrDelay')

# Start the actual modelling
print("Start Model Fitting")
start = time.time()
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=rmse, numFolds=2)
cv_model = cv.fit(flights_train)
print("Time taken to develop model: " + str(time.time()-start) + 's.')

# Save the model
cv_model.write().overwrite().save(modelPath)

# Evaluation of the best model according to evaluator
print("RMSE training: " + str(rmse.evaluate(cv_model.bestModel.transform(flights_train))))
print('RMSE test: ' + str(rmse.evaluate(cv_model.bestModel.transform(flights_test))))

# Stop Spark
spark.stop()