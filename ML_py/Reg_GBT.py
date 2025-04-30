#******GBT Model
# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *

from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressionModel, GBTRegressor
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from functools import reduce
import pandas as pd
import builtins

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

IS_DB = False # Run the code in Databricks

PYSPARK_CLI = True
if PYSPARK_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# File location and type
file_location = "/user/xtang13/project/reprojected_resampled_raster_with_indices.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","


df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

df.show(5)
#if can read, continues

#rename multiple coulmns 
df = df.withColumnRenamed("Band_1","burned") \
        .withColumnRenamed("Band_2","lst") \
        .withColumnRenamed("Band_3","humidity") \
        .withColumnRenamed("Band_4","precip") \
        .withColumnRenamed("Band_5","landcover") \
        .withColumnRenamed("Band_6","elevation") \
        .withColumnRenamed("Band_7","slope") \
        .withColumnRenamed("Band_8","aspect") \
        .withColumnRenamed("Band_9","pop_density") \
        .withColumnRenamed("row","row_px") \
        .withColumnRenamed("col","col_px") 

df.show(5)

projectSchema = StructType([
  StructField("burned", IntegerType(), False),
  StructField("lst", IntegerType(), False),
  StructField("humidity", IntegerType(), False),
  StructField("precip", IntegerType(), False),
  StructField("landcover", IntegerType(), False),
  StructField("elevation", IntegerType(), False),
  StructField("slope", IntegerType(), False),
  StructField("aspect", IntegerType(), False),
  StructField("pop_density", IntegerType(), False),
  StructField("Row", IntegerType(), False),
  StructField("Col", IntegerType(), False),
])

#if one column has 0 or null, remove the entire row 

# List of columns to check (same as in your code)
columns_to_check = [
    "lst", "humidity", "precip", "landcover",
    "elevation", "slope", "aspect", "pop_density"
]

# Build AND condition: all columns must be not null and not 0
conditions = [(col(c).isNotNull()) & (col(c) != 0) for c in columns_to_check]
combined_condition = reduce(lambda x, y: x & y, conditions)

# Filter DataFrame
df = df.filter(combined_condition)

df.show(5)

# Create a view or table
temp_table_name = "wildFire_class_GBT_csv"
df.createOrReplaceTempView(temp_table_name)

#using spark sql 
df_data_1 = spark.sql("SELECT * FROM wildFire_class_GBT_csv") #df_data_1


# Select features and label
data = df_data_1.select("lst", "humidity", "precip", "landcover", "elevation", "slope", "aspect", "pop_density", col("burned").alias("label"))

#new feature; bc of error of null value
feature_cols = ["lst", "humidity", "precip", "landcover", "elevation", "slope", "aspect", "pop_density"]

# Drop rows with nulls in feature columns
clean_data = data.dropna(subset=feature_cols)

#===============weighted column==========================
# Count how many are burned (1) and not burned (0)
majority_count = clean_data.filter(col("label") == 0).count()
minority_count = clean_data.filter(col("label") == 1).count()

balancing_ratio = majority_count / (majority_count + minority_count)

from pyspark.sql.functions import when

# Create a new column that gives higher weight to the minority class
weighted_data = clean_data.withColumn(
    "classWeightCol",
    when(col("label") == 1, balancing_ratio).otherwise(1 - balancing_ratio)
)
#===============weighted column==========================


# Split the data
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
train_rows = train.count()
test_rows = test.count()
print("Training Rows:", train_rows, " Testing Rows:", test_rows)

# Define the pipeline/prepare training 
assembler = VectorAssembler(inputCols = ["lst", "humidity", "precip", "landcover", "elevation", "slope", "aspect", "pop_density"], outputCol="features")
# minMax Scale; number vector is normalized: 04/20/2021
minMax = MinMaxScaler(inputCol = assembler.getOutputCol(), outputCol="normFeatures")

#build GBT model
gbt = GBTRegressor(labelCol="label", featuresCol="normFeatures")

# Tune Parameters
# define list of models made from Train Validation Split and Cross Validation
model = []
pipeline = []

# Create a parameter grid
paramGrid = ParamGridBuilder() \
  .addGrid(gbt.maxDepth, [2, 3]) \
  .addGrid(gbt.maxBins, [5, 10]) \
  .addGrid(gbt.minInfoGain, [0.0]) \
  .build()

pipeline.insert(0, Pipeline(stages=[assembler, minMax, gbt]))
tvs = TrainValidationSplit(estimator=pipeline[0], evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)
model.insert(0, tvs.fit(train))

# TVS model
best_pipeline_model_tvs = model[0].bestModel
best_model_tvs = best_pipeline_model_tvs.stages[-1]

importances_tvs = best_model_tvs.featureImportances.toArray()
features = assembler.getInputCols()

featureImp_tvs = pd.DataFrame(
    list(zip(features, importances_tvs)),
    columns=["feature", "importance"]
)
featureImp_tvs["abs_importance"] = featureImp_tvs["importance"].abs()
featureImp_tvs = featureImp_tvs.sort_values(by="abs_importance", ascending=False)



# Build the best model using Cross Validator:
# The combination of parameters with (maxDepth: [2]), (maxBins, [10]), (minInfoGain, [0.0]), which is the same as the first model above
paramGridCV = ParamGridBuilder() \
  .addGrid(gbt.maxDepth, [2]) \
  .addGrid(gbt.maxBins, [10]) \
  .addGrid(gbt.minInfoGain, [0.0]) \
  .build()

# Complete the Cross Validator
pipeline.insert(1, Pipeline(stages=[assembler, minMax, gbt]))
# K=3, 5
K = 3
cv = CrossValidator(estimator=pipeline[1], evaluator=RegressionEvaluator(), estimatorParamMaps=paramGridCV, numFolds=K)
# the second best model
model.insert(1, cv.fit(train))

# CV model
best_pipeline_model_cv = model[1].bestModel
best_model_cv = best_pipeline_model_cv.stages[-1]

importances_cv = best_model_cv.featureImportances.toArray()
# (features are already defined above)

featureImp_cv = pd.DataFrame(
    list(zip(features, importances_cv)),
    columns=["feature", "importance"]
)
featureImp_cv["abs_importance"] = featureImp_cv["importance"].abs()
featureImp_cv = featureImp_cv.sort_values(by="abs_importance", ascending=False)

# Print the feature importances
print("=== Feature Importances from TrainValidationSplit (TVS) ===")
print(featureImp_tvs)

print("\n=== Feature Importances from CrossValidator (CV) ===")
print(featureImp_cv)


### Test the Model
# list prediction
prediction = []
predicted = []

for i in range(2):
  prediction.insert(i, model[i].transform(test))


#examine predicted & actual values

for i in range(2):
  predicted.insert(i, prediction[i].select("normFeatures", "prediction", "trueLabel"))
  predicted[i].show(20)

### Retrieve the Root Mean Square Error (RMSE)
rmses = []
for i in range(2):
  evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
  rmse = evaluator.evaluate(predicted[i])
  rmses.insert(i, rmse)
  print ("Model ", i, ": ", "Root Mean Square Error (RMSE):", rmses[i])

#calculate R2
r2s = []
for i in range(2):
  evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="r2")
  r2 = evaluator.evaluate(predicted[i])
  r2s.insert(i, r2)
  print ("Model ", i, ": ", "Coefficient of Determination (R2):", r2s[i])



'''
TVS
25/04/26 20:59:59 INFO DAGScheduler: Job 474 finished: treeAggregate at Statistics.scala:58, took 7.096884 s
Model  0 :  Root Mean Square Error (RMSE): 0.21048852442958974
CV
25/04/26 21:00:06 INFO DAGScheduler: Job 475 finished: treeAggregate at Statistics.scala:58, took 6.568748 s
Model  1 :  Root Mean Square Error (RMSE): 0.21669416323216747

TVS
25/04/26 21:00:13 INFO DAGScheduler: Job 476 finished: treeAggregate at Statistics.scala:58, took 7.009866 s
Model  0 :  Coefficient of Determination (R2): 0.2715590437174674
CV
25/04/26 21:00:20 INFO DAGScheduler: Job 477 finished: treeAggregate at Statistics.scala:58, took 6.817395 s
Model  1 :  Coefficient of Determination (R2): 0.22797398389632684

=== Feature Importances from TrainValidationSplit (TVS) ===
       feature  importance  abs_importance
2       precip    0.205551        0.205551
0          lst    0.195445        0.195445
3    landcover    0.193150        0.193150
4    elevation    0.177831        0.177831
1     humidity    0.119499        0.119499
5        slope    0.074444        0.074444
7  pop_density    0.022944        0.022944
6       aspect    0.011135        0.011135

=== Feature Importances from CrossValidator (CV) ===
       feature  importance  abs_importance
3    landcover    0.388587        0.388587
4    elevation    0.263370        0.263370
2       precip    0.188226        0.188226
5        slope    0.080365        0.080365
1     humidity    0.034844        0.034844
7  pop_density    0.034522        0.034522
0          lst    0.010086        0.010086
6       aspect    0.000000        0.000000
'''














