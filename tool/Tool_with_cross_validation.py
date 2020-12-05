# Import all needed functions
import time
import sys
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator
from pyspark.sql.types import IntegerType

# SET UP OF ENVIRONMENT
# Create a Spark Session and check its version
spark = SparkSession.builder.appName('Delay Classifier').master('local[*]').getOrCreate()
print("Spark version", spark.version)

# Path to the file
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
lr = LinearRegression(featuresCol= 'features', labelCol='ArrDelay')
rmse = RegressionEvaluator(labelCol='ArrDelay')

# Hyperparameter tuning (using gridsearch)
params = ParamGridBuilder().addGrid(lr.regParam, [0.01]) \
                        .addGrid(lr.elasticNetParam, [0.5])\
                        .build()

# Construct a pipeline
pipeline = Pipeline(stages=[indexer, encoder, assembler, lr])

# Modelling
print("Start Model Fitting")
start = time.time()
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=rmse, numFolds=5)
cv_model = cv.fit(flights_train)
print("Time taken to develop model: " + str(time.time()-start) + 's.')

modelPath = "/lr_model"
cv_model.save(modelPath)

# Evaluation of the best model according to evaluator
print("RMSE training: " + str(rmse.evaluate(cv_model.bestModel.transform(flights_train))))
print('RMSE test: ' + str(rmse.evaluate(cv_model.bestModel.transform(flights_test))))


# if __name__ == "__main__":
#     print('Hello')