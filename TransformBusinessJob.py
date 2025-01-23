import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, from_json, struct
from pyspark.sql.types import *

# Parse job arguments
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

# Initialize Spark and Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Define schema for the JSON data
schema = StructType([
    StructField("business_id", StringType(), True),
    StructField("name", StringType(), True),
    StructField("address", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("postal_code", StringType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("stars", DoubleType(), True),
    StructField("review_count", IntegerType(), True),
    StructField("is_open", IntegerType(), True),
    StructField("attributes", MapType(StringType(), StringType()), True),
    StructField("categories", StringType(), True),
    StructField("hours", MapType(StringType(), StringType()), True)
])

# Your specific S3 paths
input_path = "s3://yelp-raw-data/yelp_academic_dataset_business.json"
output_path = "s3://yelp-raw-data/Output/business_data_consolidated.parquet"

# Read JSON data
df = spark.read.json(input_path, schema=schema)

# Transformation Steps:
# 1. Select and reorder columns
transformed_data = df.select(
    "business_id",
    "name",
    "address",
    "city",
    "state",
    "postal_code",
    "latitude",
    "longitude",
    "stars",
    "review_count",
    "is_open",
    "categories"
)

# 2. Remove NULL Values in critical columns
columns_to_check_for_null = [
    'business_id', 
    'name', 
    'city', 
    'state', 
    'postal_code', 
    'latitude', 
    'longitude'
]
transformed_data = transformed_data.dropna(subset=columns_to_check_for_null)

# 3. Remove duplicates based on business_id
transformed_data = transformed_data.dropDuplicates(["business_id"])

# 4. Write as a single parquet file
transformed_data.coalesce(1).write.mode("overwrite").parquet(output_path)

# Print the count of processed records
print(f"Total records processed: {transformed_data.count()}")

# Commit the job
job.commit()