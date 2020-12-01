# Import all needed functions
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator
from pyspark.sql.types import IntegerType

# SET UP OF ENVIRONMENT
# Create a Spark Session
spark = SparkSession.builder.getOrCreate()

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
                    .withColumn('Distance', airports['Distance'].cast(IntegerType()))
                    #.withColumn('DayOfWeek', airports['DayOfWeek'].cast(IntegerType())) \
                    #.withColumn('DayOfMonth', airports['DayOfMonth'].cast(IntegerType())) \
                    #.withColumn('DepTime', airports['DepTime'].cast(IntegerType())) \

#airports.printSchema()

# Filter all the rows where ArrDelay is unknown
airports = airports.where(airports.ArrDelay.isNotNull())

# FEATURE SELECTION
assembler = VectorAssembler(inputCols=[
    'DepDelay', 'Distance'
], outputCol='features')

# MACHINE LEARNING
airports_prep = assembler.transform(airports)
airports_final = airports_prep.select(['features',  'ArrDelay'])

# Split data
flights_train, flights_test = airports_final.randomSplit([0.9, 0.1], seed=123)
flights_train.show()

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
pipeline = Pipeline(stages=[lr])

cv = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=rmse, numFolds=5)
cv_model = cv.fit(flights_train)

print("RMSE training:" + str(cv_model.summary.rootMeanSquaredError))
print('RMSE test:' + str(rmse.evaluate(cv_model.transform(flights_test))))
