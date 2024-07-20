from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, LongType, FloatType

# Initialize Spark session
spark = SparkSession.builder.appName('StudentDataProcessing').getOrCreate()

# Define schema for incoming data
schema = StructType([
    StructField('student_id', IntegerType(), True),
    StructField('timestamp', LongType(), True),
    StructField('activity', StringType(), True),
    StructField('content_id', IntegerType(), True),
    StructField('score', FloatType(), True)
])

# Read data from Kafka
df = spark.readStream.format('kafka')\
    .option('kafka.bootstrap.servers', 'localhost:9092')\
    .option('subscribe', 'student_data')\
    .load()

# Parse the JSON data
df = df.selectExpr("CAST(value AS STRING)")
df = df.withColumn('data', from_json(col('value'), schema)).select('data.*')

# Write the data to MongoDB
df.writeStream \
    .foreachBatch(lambda batch_df, _: batch_df.write.format('mongo')
                  .option('uri', 'mongodb://localhost:27017/education_system.student_interactions')
                  .mode('append')
                  .save()) \
    .start()
