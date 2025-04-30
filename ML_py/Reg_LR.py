#******Linear Regression
# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *

from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
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
temp_table_name = "wildFire_regression_LR_csv"
df.createOrReplaceTempView(temp_table_name)

#using spark sql 
df_data_1 = spark.sql("SELECT * FROM wildFire_regression_LR_csv") #df_data_1

# Show the dataframe's datatypes
df_data_1.dtypes

#assign varibale df_data_x to rdd_from_df
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

#this split is important 2 filtering session; due to encountered error of null values 
#split data 
splits = weighted_data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1]
train_rows = train.count()
test_rows = test.count()
print("Training Rows:", train_rows, " Testing Rows:", test_rows)


# Define the pipeline/prepare training 
assembler = VectorAssembler(inputCols = ["lst", "humidity", "precip", "landcover", "elevation", "slope", "aspect", "pop_density"], outputCol="features")
# minMax Scale; number vector is normalized: 04/20/2021
minMax = MinMaxScaler(inputCol = assembler.getOutputCol(), outputCol="normFeatures")

# build LR model 
lr = LinearRegression(labelCol="label",featuresCol="normFeatures", maxIter=10, regParam=0.3)

# Tune Parameters
# define list of models made from Train Validation Split and Cross Validation
model = []
pipeline = []

# Create a parameter grid
paramGrid = ParamGridBuilder() \
  .addGrid(lr.maxIter, [2, 3]) \
  .addGrid(lr.regParam, [0.01, 0.03]) \
  .build()

# Create a pipeline
pipeline = Pipeline(stages=[assembler, minMax, lr]) 

# Create a TrainValidationSplit
tvs = TrainValidationSplit(
    estimator=pipeline,  # ✅ Pass the actual pipeline
    estimatorParamMaps=paramGrid,
    evaluator=RegressionEvaluator(labelCol="label", predictionCol="prediction"),
    trainRatio=0.8
)

# Store the trained model in a list
model.insert(0, tvs.fit(train))


# Build the best model using Cross Validator:
pipeline = Pipeline(stages=[assembler, minMax, lr])
k = 3
cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=RegressionEvaluator(labelCol="label", predictionCol="prediction"),
    numFolds=k
)

# Fit the model
model.insert(1, cv.fit(train))  # Use .insert() to add at index 1

# From TrainValidationSplit
tvs_best_model = model[0].bestModel.stages[-1]  # Get LinearRegressionModel
tvs_coefficients = tvs_best_model.coefficients.toArray()

# From CrossValidator
cv_best_model = model[1].bestModel.stages[-1]    # Get LinearRegressionModel
cv_coefficients = cv_best_model.coefficients.toArray()

# Get feature names
features = assembler.getInputCols()

# Create DataFrames
tvs_featureImp = pd.DataFrame(
    list(zip(features, tvs_coefficients)),
    columns=["feature", "coefficient"]
)
tvs_featureImp["abs_importance"] = tvs_featureImp["coefficient"].abs()
tvs_featureImp = tvs_featureImp.sort_values(by="abs_importance", ascending=False)

cv_featureImp = pd.DataFrame(
    list(zip(features, cv_coefficients)),
    columns=["feature", "coefficient"]
)
cv_featureImp["abs_importance"] = cv_featureImp["coefficient"].abs()
cv_featureImp = cv_featureImp.sort_values(by="abs_importance", ascending=False)

print("=== TrainValidationSplit Feature Importances ===")
print(tvs_featureImp)

print("\n=== CrossValidator Feature Importances ===")
print(cv_featureImp)


# Test the Model
prediction = []
predicted = []

# predict based on the test data
for i in range(2):
    pred = model[i].transform(test)
    pred = pred.withColumnRenamed("label", "trueLabel")  # immediately rename after prediction
    prediction.insert(i, pred)

# examine predicted & actual values
for i in range(2):
    predicted.insert(i, prediction[i].select("normFeatures", "prediction", "trueLabel"))
    predicted[i].show(20)

# Retrieve the Root Mean Square Error (RMSE) and R2
rmses = []
r2s = []

# Create evaluators once
evaluator_rmse = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="r2")

for i in range(2):
    rmse = evaluator_rmse.evaluate(predicted[i])
    r2 = evaluator_r2.evaluate(predicted[i])
    rmses.append(rmse)
    r2s.append(r2)
    print(f"Model {i}: RMSE = {rmse:.4f}, R2 = {r2:.4f}")



'''
=== TrainValidationSplit Feature Importances ===
       feature  coefficient  abs_importance
2       precip     0.347238        0.347238
7  pop_density     0.323382        0.323382
3    landcover     0.309030        0.309030
5        slope     0.198257        0.198257
4    elevation     0.171926        0.171926
0          lst     0.167877        0.167877
1     humidity     0.146250        0.146250
6       aspect    -0.011826        0.011826

=== CrossValidator Feature Importances ===
       feature  coefficient  abs_importance
2       precip     0.347238        0.347238
7  pop_density     0.323382        0.323382
3    landcover     0.309030        0.309030
5        slope     0.198257        0.198257
4    elevation     0.171926        0.171926
0          lst     0.167877        0.167877
1     humidity     0.146250        0.146250
6       aspect    -0.011826        0.011826

scala:58, took 5.980611 s
Model 0: RMSE = 0.2361, R2 = 0.0841

scala:58, took 6.166839 s
Model 1: RMSE = 0.2361, R2 = 0.0841

'''
