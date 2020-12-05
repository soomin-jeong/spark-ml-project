# Import all needed functions
import time
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator
from pyspark.sql.types import IntegerType

# SET UP OF ENVIRONMENT
# Create a Spark Session
spark = SparkSession.builder.appName('Delay Classifier').master('local[*]').getOrCreate()

# Print the version
print("Spark version", spark.version)

# Path to the file
file_path = "1990.csv.bz2"

# Read in the airports data
airports = spark.read.csv(file_path, header=True)
airports = airports.drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay",
                         "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")

# Show the data
# airports.show(3)
print("Data loaded")

# FEATURE ENGINGEERING
airports = airports.withColumn('CRSDepTime', airports['CRSDepTime'].cast(IntegerType())) \
                    .withColumn('ArrDelay', airports['ArrDelay'].cast(IntegerType())) \
                    .withColumn('DepDelay', airports['DepDelay'].cast(IntegerType())) \
                    .withColumn('Distance', airports['Distance'].cast(IntegerType())) \
                    .withColumn('DayOfWeek', airports['DayOfWeek'].cast(IntegerType())) \
                    .withColumn('Month', airports['Month'].cast(IntegerType())) \
                    #.withColumn('DepTime', airports['DepTime'].cast(IntegerType())) \

#airports.printSchema()

# Transform strings to categories
indexer = StringIndexer(inputCols=['UniqueCarrier'], outputCols=['carrierIndex'])

# One hot encoding
encoder = OneHotEncoder(inputCols=['DayOfWeek', 'Month', 'carrierIndex'],
                        outputCols=['DayOfWeekVec', 'MonthVec', 'carrierVec'])

# Filter all the rows where ArrDelay is unknown
airports = airports.where(airports.ArrDelay.isNotNull())
# Split data
flights_train, flights_test = airports.randomSplit([0.9, 0.1], seed=123)
flights_train.show()

# FEATURE SELECTION
assembler = VectorAssembler(inputCols=[
    'DepDelay', 'DayOfWeekVec', 'MonthVec', 'carrierVec'
], outputCol='features')

# MACHINE LEARNING
# Regressor Choice
# Maybe an idea to let user choose different algorithms
lr = LinearRegression(featuresCol= 'features', labelCol='ArrDelay')
rmse = RegressionEvaluator(labelCol='ArrDelay')

# Include Hyperparameter tuning (using gridsearch or different)
# Create parameter grid
params = ParamGridBuilder()

# Add grids for two parameters
params = params.addGrid(lr.regParam, [0.01]) \
               .addGrid(lr.elasticNetParam, [0.5])

# Build the parameter grid
params = params.build()

# Construct a pipeline
pipeline = Pipeline(stages=[indexer, encoder, assembler, lr])

print("Start Model Fitting")
start = time.time()
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=rmse, numFolds=5)
cv_model = cv.fit(flights_train)
print("Time taken to develop model: " + str(time.time()-start) + 's.')


print("RMSE training: " + str(rmse.evaluate(cv_model.bestModel.transform(flights_train))))
print('RMSE test: ' + str(rmse.evaluate(cv_model.bestModel.transform(flights_test))))
