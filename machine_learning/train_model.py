from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName('StudentPerformancePrediction').getOrCreate()

# Load data from MongoDB
df = spark.read.format('mongo')\
    .option('uri', 'mongodb://localhost:27017/education_system.student_interactions')\
    .load()

# Feature extraction
assembler = VectorAssembler(inputCols=['student_id', 'content_id', 'score'], outputCol='features')
data = assembler.transform(df)

# Add a column for predicting future performance (simulated data)
data = data.withColumn('label', col('score'))

# Train-test split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

# Train a Random Forest model
rf = RandomForestRegressor(featuresCol='features', labelCol='label')
rf_model = rf.fit(train_data)

# Evaluate the model
predictions = rf_model.transform(test_data)
evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction', metricName='rmse')
rmse = evaluator.evaluate(predictions)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Save the model
rf_model.save('models/student_performance_model')
