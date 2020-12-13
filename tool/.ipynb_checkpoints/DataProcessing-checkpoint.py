import time
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
import pyspark.sql.functions as f

from pyspark.sql import Row as row
from pyspark.sql import DataFrameStatFunctions as statFunc

# Outliers:
# from py.path import local


# SET UP OF ENVIRONMENT
# Create a Spark Session and check its version:
# Create a Spark Session and check its version
from pyspark.sql.types import IntegerType

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

# Introduce index column to perform Exploratory Analysis:

airports2 = airports.select("*").withColumn("id", monotonically_increasing_id())
airports2.show(n=10)

# Transform string to integer to perform the analysis:
airports2 = airports2.withColumn('CRSDepTime', airports2['CRSDepTime'].cast(IntegerType())) \
                    .withColumn('ArrDelay', airports2['ArrDelay'].cast(IntegerType())) \
                    .withColumn('DepDelay', airports2['DepDelay'].cast(IntegerType())) \
                    .withColumn('Distance', airports2['Distance'].cast(IntegerType())) \
                    .withColumn('DayOfWeek', airports2['DayOfWeek'].cast(IntegerType())) \
                    .withColumn('Month', airports2['Month'].cast(IntegerType())) \

# HANDLING DUPLICATE VALUES:

# Delete all cases with identical data: (all the line duplicated):

# Check if the lines are duplicated:
airports2.count(), airports2.distinct().count()

# if all are duplicated, remove them:
airports3 = airports2.dropDuplicates()
print(" Delete all cases with identical data: (all the line duplicated)")

{
    airports2
    .groupby(airports2.columns)
    .count()
    .filter('count > 1')
    .show(10)
}

# MANAGE MISSING VALUES:
print("----------------- MANAGE MISSING VALUES ----------------")
print("Calculate the number of missing observations in a row: ")

# Calculate the number of missing observtions in a row:
(
    spark.createDataFrame(
        airports3.rdd.map(
            lambda row: (
                row['id'],
                sum([c == "NA" for c in row])
            )
        )
        .collect(),
        ['id', 'MissingValues']
    )
    .orderBy('MissingValues', ascending=False)
    .show(10)
)

# with this missing values then we can see what strategy we want to perform with them.
# DISCUSS WITH SOO AND TOM.




# HANDLING OUTLIERS:


print("------------------------ HANDLING OUTLIERS ---------------------------------")

# Choose the features:

# Year can not be supported yet at not be in string form (integrate with Tom's script)
features = ['CRSDepTime', 'ArrDelay', 'DepDelay', 'Distance', 'DayOfWeek', 'Month']

quantiles = [0.25, 0.75]

cutOffPoints = []

# Rule for determine outlier element:

for feature in features:
    qts = airports3.approxQuantile(feature, quantiles, 0.05)
    # Calculating the first and third quantile by method cutOffPoints
    # IQR = Interquartil Range

    IQR = qts[1] - qts[0]
    cutOffPoints.append((feature, [
        qts[0] - 1.5 * IQR,
        qts[1] + 1.5 * IQR,
    ]))
    cutOffPoints = dict(cutOffPoints)

# Report outliers:

aberrant_value = airports3.select(*['id'] + [
    (
        (airports3[f] < cutOffPoints[f][0]) |
        (airports3[f] > cutOffPoints[f][1])
    ).alias(f + '_b') for f in features
])

print("Outliers Found: " + aberrant_value.show())

# NEXT STEPS Exploratory Analysis:
# - Perform descriptive statistics.
# - Perform correlation tests.
# - Perform PCA Analysis.

# NEXT STEPS Machine Learning Models:
# - Linear regression algorithm.
# - Classification.
# - Logistic regression.
# - Random forest.
# - K-means clustering. 