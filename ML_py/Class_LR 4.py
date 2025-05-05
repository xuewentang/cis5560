#scp /Users/sophathriya/Desktop/Class_LR.py ssen5@144.24.13.0:~

#******LogisticRegression (classification)
# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import col
from functools import reduce
import pandas as pd
import builtins
from time import time 

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

#just making sure the field types are correct
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
temp_table_name = "wildFire_class_LR_csv"
df.createOrReplaceTempView(temp_table_name)

#using spark sql 
df_data_2 = spark.sql("SELECT * FROM wildFire_class_LR_csv") #df_data_1

# Show the dataframe's datatypes
df_data_2.dtypes

#assign varibale df_data_x to rdd_from_df
data = df_data_2.select("lst", "humidity", "precip", "landcover", "elevation", "slope", "aspect", "pop_density", col("burned").alias("label"))


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

#split data 
start = time()
splits = weighted_data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1]
train_rows = train.count()
test_rows = test.count()
print("Training Rows:", train_rows, " Testing Rows:", test_rows)

#prepare training data 
assembler = VectorAssembler(inputCols = ["lst", "humidity", "precip", "landcover", "elevation", "slope", "aspect", "pop_density"], outputCol="features")
# minMax Scale; number vector is normalized: 04/20/2021
minMax = MinMaxScaler(inputCol = assembler.getOutputCol(), outputCol="normFeatures")

training = assembler.transform(train)
training = minMax.fit(training).transform(training)
training = training.select("normFeatures", "label", "classWeightCol") #no rename


#train Linear Regression model 
lr = LogisticRegression(
    labelCol="label",
    featuresCol="normFeatures",
    weightCol="classWeightCol", 
    maxIter=10,
    regParam=0.3
)

# define list of models made from Train Validation Split and Cross Validation
model = []
pipeline = []

# Set up the parameter grid 
#====================uncomment the parameter and it will run for more than 3 hours==================== 
paramGridTV = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 0.3]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()
    #.addGrid(lr.maxIter, [10, 50, 100]) \
    #.addGrid(lr.threshold, [0.4, 0.5, 0.6]) \
    


# Set up the evaluator for TrainValidator
evaluator_TV = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

pipeline.insert(0, Pipeline(stages=[assembler, minMax, lr]))
# Set up TrainValidationSplit
tvs = TrainValidationSplit(
    estimator=pipeline[0],
    estimatorParamMaps=paramGridTV,
    evaluator=evaluator_TV,
    trainRatio=0.8  # 80% train, 20% validation
)
#train the model 
model.insert(0, tvs.fit(train))
print("TrainValidationSplit model trained!")


#extract best model 
lrModel_tvs = model[0].bestModel
best_lr_model_tvs = lrModel_tvs.stages[-1]

#extract feature importance 
coefficients_tvs = best_lr_model_tvs.coefficients.toArray()
intercept_tv = best_lr_model_tvs.intercept

featureImp_tvs = pd.DataFrame(
    list(zip(assembler.getInputCols(), coefficients_tvs)),
    columns=["feature", "coefficient"]
)

# Calculate absolute importance for easier sorting
featureImp_tvs["abs_importance"] = featureImp_tvs["coefficient"].apply(lambda x: abs(x))
#sort 
featureImp_tvs = featureImp_tvs.sort_values(by="abs_importance", ascending=False)


#=======================TVS ends here 

#set up evaluator for CrossValidator
paramGridCV = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .addGrid(lr.maxIter, [50, 100]) \
    .build()
    # .addGrid(lr.threshold, [0.4, 0.5, 0.6]) \
# Set up the evaluator for CrossValidator
evaluator_CV = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)


pipeline.insert(1, Pipeline(stages=[assembler, minMax, lr]))
# Set up CrossValidator
cv = CrossValidator(
    estimator=pipeline[1],
    estimatorParamMaps=paramGridCV,
    evaluator=evaluator_CV,
    numFolds=5,   # 5-fold cross-validation
    #parallelism=2 # parallel training (optional)
)
model.insert(1, cv.fit(train))
print("CrossValidator model trained!")


#extract best model 
best_pipeline_model_cv = model[1].bestModel
best_lr_model_cv = best_pipeline_model_cv.stages[-1]

#extract feature importance 
coefficients_cv = best_lr_model_cv.coefficients.toArray()
intercept_cv = best_lr_model_cv.intercept

featureImp_tvs = pd.DataFrame(
    list(zip(assembler.getInputCols(), coefficients_cv)),
    columns=["feature", "coefficient"]
)

print("\n=== Feature Importances (CrossValidator Model) ===")
best_pipeline_model_cv = model[1].bestModel
best_lr_model_cv = best_pipeline_model_cv.stages[-1]


featureImp_cv = pd.DataFrame(
    list(zip(assembler.getInputCols(), coefficients_cv)),
    columns=["feature", "coefficient"]
)


# 1. Assemble train features (to fit scaler correctly)
train_assembled = assembler.transform(train)

# 2. Fit the scaler on "features"
scaler_model = minMax.fit(train_assembled)

# 3. Assemble test features
testing = assembler.transform(test)

# 4. Apply scaler to test
testing = scaler_model.transform(testing)

# 5. Select correct columns for model input
testing = testing.select(
    col("normFeatures"),
    col("label").alias("trueLabel")
)


# have a list of models we have ([tvs_model [0], cv_model[1]])
# And a list of names for these models
model_names = ["TrainValidationSplit", "CrossValidator"]

precisions = []
recalls = []
f1s = []
accuracies = []
aucs = []

for idx, m in enumerate(model):
    print(f"\n=== Feature Importances ({model_names[idx]} Model) ===")
    
    # Extract best logistic regression model
    best_model = m.bestModel
    if hasattr(best_model, "stages"):
        lr_model = best_model.stages[-1]
    else:
        lr_model = best_model

    # ✅ First extract coefficients
    coefficients = lr_model.coefficients.toArray()

    # ✅ Build feature importance DataFrame
    feature_imp = pd.DataFrame(
        list(zip(assembler.getInputCols(), coefficients)),
        columns=["feature", "coefficient"]
    )

    # ✅ Add absolute importance
    feature_imp["abs_importance"] = feature_imp["coefficient"].apply(lambda x: abs(x))

    # ✅ Then sort
    feature_imp = feature_imp.sort_values(by="abs_importance", ascending=False)

    # ✅ Now you can print it
    print(feature_imp.round(4))

    # ✅ After printing, continue with prediction
    prediction = lr_model.transform(testing)


    # Confusion matrix calculations
    tp = float(prediction.filter("prediction == 1.0 AND trueLabel == 1.0").count())
    fp = float(prediction.filter("prediction == 1.0 AND trueLabel == 0.0").count())
    tn = float(prediction.filter("prediction == 0.0 AND trueLabel == 0.0").count())
    fn = float(prediction.filter("prediction == 0.0 AND trueLabel == 1.0").count())

    print(f"\nConfusion Matrix for {model_names[idx]} Model:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")

    precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_manual = 2 * (precision_manual * recall_manual) / (precision_manual + recall_manual) if (precision_manual + recall_manual) > 0 else 0.0
    accuracy_manual = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    auc = BinaryClassificationEvaluator(
        labelCol="trueLabel",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    ).evaluate(prediction)

    precisions.append(precision_manual)
    recalls.append(recall_manual)
    f1s.append(f1_manual)
    accuracies.append(accuracy_manual)
    aucs.append(auc)

# After loop, build summary table
summary_df = pd.DataFrame({
    "Model": model_names,
    "Precision": precisions,
    "Recall": recalls,
    "F1 Score": f1s,
    "Accuracy": accuracies,
    "AUC": aucs
}).round(4)

print("\n=== Model Performance Summary ===")
print(summary_df)





'''

=== Feature Importances (TrainValidationSplit Model) ===
       feature  coefficient  abs_importance
7  pop_density      -4.5496          4.5496
3    landcover       3.9629          3.9629
2       precip       2.8658          2.8658
5        slope       1.9170          1.9170
4    elevation       1.5061          1.5061
1     humidity       1.0628          1.0628
0          lst      -0.5277          0.5277
6       aspect      -0.0797          0.0797

Confusion Matrix for TrainValidationSplit Model:
True Positives (TP): 77105.0
False Positives (FP): 397826.0
True Negatives (TN): 1095210.0
False Negatives (FN): 26842.0

=== Feature Importances (CrossValidator Model) ===
       feature  coefficient  abs_importance
7  pop_density     -12.7606         12.7606
3    landcover       5.5140          5.5140
2       precip       5.1568          5.1568
4    elevation       3.4171          3.4171
1     humidity       3.3550          3.3550
5        slope       2.9371          2.9371
0          lst       1.6506          1.6506
6       aspect      -0.1566          0.1566

Confusion Matrix for CrossValidator Model:
True Positives (TP): 77837.0
False Positives (FP): 367393.0
True Negatives (TN): 1125643.0
False Negatives (FN): 26110.0

=== Model Performance Summary ===
                  Model  Precision  Recall  F1 Score  Accuracy     AUC
0  TrainValidationSplit     0.1623  0.7418    0.2664    0.7341  0.8149
1        CrossValidator     0.1748  0.7488    0.2835    0.7536  0.822125

'''


