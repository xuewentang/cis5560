#scp /Users/sophathriya/Desktop/Class_RFR.py ssen5@144.24.13.0:~


#******RandomForestClassificationModel
# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import RandomForestClassifier

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
file_location = "/user/ssen5/reprojected_resampled_raster_with_indices.csv"
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
        .withColumnRenamed("row","longtitude") \
        .withColumnRenamed("col","latitude") 

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
'''
#filter out rows is 0 or null 
df2 = df.filter(
    (col("lst").isNotNull()) & (col("lst") != 0) &
    (col("humidity").isNotNull()) & (col("humidity") != 0) &
    (col("precip").isNotNull()) & (col("precip") != 0) &
    (col("landcover").isNotNull()) & (col("landcover") != 0) &
    (col("elevation").isNotNull()) & (col("elevation") != 0) &
    (col("slope").isNotNull()) & (col("slope") != 0) &
    (col("aspect").isNotNull()) & (col("aspect") != 0) &
    (col("pop_density").isNotNull()) & (col("pop_density") != 0)
)
df2.show(5)
'''

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
temp_table_name = "wildFire_class_RF_csv"
df.createOrReplaceTempView(temp_table_name)

#using spark sql 
df_data_2 = spark.sql("SELECT * FROM wildFire_class_RF_csv") #df_data_1


# Select features and label
data = df.select("lst", "humidity", "precip", "landcover", "elevation", "slope", "aspect", "pop_density", col("burned").alias("label"))

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

#if results show change line 146 to splits = weighted_data.randomSplit([0.7, 0.3])
# Split the data
splits = weighted_data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1]
train_rows = train.count()
test_rows = test.count()
print("Training Rows:", train_rows, " Testing Rows:", test_rows)

# Define the pipeline/prepare training 
assembler = VectorAssembler(inputCols = ["lst", "humidity", "precip", "landcover", "elevation", "slope", "aspect", "pop_density"], outputCol="features")
# minMax Scale; number vector is normalized: 04/20/2021
#minMax = MinMaxScaler(inputCol = assembler.getOutputCol(), outputCol="normFeatures")

# define list of models made from Train Validation Split and Cross Validation
model = []
pipeline = []

#lr = RandomForestRegressor(labelCol="label", featuresCol="normFeatures")
rf = RandomForestClassifier(labelCol="label", featuresCol="features", weightCol="classWeightCol")
# Parameter Grid for TrainValidationSplit/ Define the parameter grid for hyperparameter tuning
paramGrid = ParamGridBuilder() \
  .addGrid(rf.maxDepth, [5, 10]) \
  .addGrid(rf.numTrees, [20, 50])\
  .build()
pipeline.insert(0, Pipeline(stages=[assembler, rf]))
tv = TrainValidationSplit(estimator=pipeline[0], evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"), estimatorParamMaps=paramGrid, trainRatio=0.8)
model.insert(0, tv.fit(train)) 
print("Train Validation Split model trained!")

# TVS Feature Importance
best_model_trainVal = model[0].bestModel.stages[-1]
feature_importance_trainVal = best_model_trainVal.featureImportances.toArray()

#Create the DataFrame
feature_imp_trainVal = pd.DataFrame(
    list(zip(assembler.getInputCols(), feature_importance_trainVal)),
    columns=["feature", "importance"]
)

# Sort it
feature_imp_trainVal = feature_imp_trainVal.sort_values(by="importance", ascending=False)

#Print it
print("Feature Importance (TrainValidationSplit):")
print(feature_imp_trainVal)


#===========cross validator with parameter
#build model using cross validator

paramGridCV = ParamGridBuilder() \
  .addGrid(rf.maxDepth, [5, 10, 15]) \
  .addGrid(rf.numTrees, [10, 30, 50]) \
  .build()
pipeline.insert(1, Pipeline(stages=[assembler, rf]))

# TODO: K = 3
# K=3, 5
K = 3
cv = CrossValidator(estimator=pipeline[1], evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"), estimatorParamMaps=paramGridCV, numFolds=K)
model.insert(1, cv.fit(train)) # Append the model to the list
print("CrossValidator model trained!")

# CV Feature Importance
best_model_crossVal = model[1].bestModel.stages[-1]
feature_importance_crossVal = best_model_crossVal.featureImportances.toArray()

feature_imp_crossVal = pd.DataFrame(
    list(zip(assembler.getInputCols(), feature_importance_crossVal)),
    columns=["feature", "importance"]
).sort_values(by="importance", ascending=False)

print("Feature Importance (CrossValidator):")
print(feature_imp_crossVal)

# 2. Predict and store results
prediction = []
predicted = []
for i in range(len(model)):
    prediction.append(model[i].transform(test))
    predicted.append(prediction[i].select("features", "prediction", "rawPrediction", "label"))

#evaluate prediction from train models

for i in range(len(model)):
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1 = evaluator_f1.evaluate(predicted[i])
    
    evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="precisionByLabel")
    precision = evaluator_precision.evaluate(predicted[i])
    
    evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="recallByLabel")
    recall = evaluator_recall.evaluate(predicted[i])
    
    evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
    auc = evaluator_auc.evaluate(predicted[i])
    
    print(f"Model {i}:")
    print(f"   F1 Score: {f1:.2%}")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall: {recall:.2%}")
    print(f"   AUC: {auc:.2%}")


'''

Train Validation Split model trained!
Feature Importance (TrainValidationSplit):
       feature  importance
3    landcover    0.302987
2       precip    0.227423
4    elevation    0.212575
1     humidity    0.155191
5        slope    0.044641
0          lst    0.029808
7  pop_density    0.023677
6       aspect    0.003698

scala:44, took 0.037942 s
Model 0:
   F1 Score: 93.02%
   Precision: 99.78%
   Recall: 91.22%
   AUC: 98.30%



CrossValidator model trained!
Feature Importance (CrossValidator):
       feature  importance
3    landcover    0.195154
2       precip    0.188120
4    elevation    0.176380
1     humidity    0.139719
7  pop_density    0.087623
0          lst    0.084224
6       aspect    0.070638
5        slope    0.058143

Model 1:
   F1 Score: 97.36%
   Precision: 100.00%
   Recall: 96.93%
   AUC: 99.95%

'''



