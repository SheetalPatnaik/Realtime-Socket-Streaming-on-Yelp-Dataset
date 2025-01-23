
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, from_json, struct
from pyspark.sql.types import *
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Parse job arguments
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

# Initialize Spark and Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Log job start
logger.info(f"Starting Glue job: {args['JOB_NAME']}")

# Define schema for the JSON data
schema = StructType([
    StructField("review_id", StringType()),
        StructField("user_id", StringType()),
        StructField("business_id", StringType()),
        StructField("stars", FloatType()),
        StructField("date", StringType()),
        StructField("text", StringType())
])

# Your specific S3 paths
input_path = "s3://yelp-raw-data/yelp_academic_dataset_review.json"
output_path = "s3://yelp-raw-data/Output/reviews/review_data_consolidated.parquet"

# Log input/output paths
logger.info(f"Reading data from: {input_path}")
logger.info(f"Writing output to: {output_path}")

# Read JSON data
df = spark.read.json(input_path, schema=schema)
logger.info(f"Initially loaded {df.count()} records")

# Transformation Steps:
# 1. Select and reorder columns
transformed_data = df.select(
    "review_id",
    "user_id",
    "business_id",
    "stars",
    "date",
    "text"
)
logger.info("Column selection completed")

# Uncomment if needed:
# 2. Remove NULL Values in critical columns
columns_to_check_for_null = [
     "review_id",
    "user_id",
    "business_id",
    "stars",
    "date",
    "text"
]
initial_count = transformed_data.count()
transformed_data = transformed_data.dropna(subset=columns_to_check_for_null)
final_count = transformed_data.count()
logger.info(f"Removed {initial_count - final_count} records with NULL values")

# 3. Remove duplicates based on business_id
initial_count = transformed_data.count()
transformed_data = transformed_data.dropDuplicates(["review_id"])
final_count = transformed_data.count()
logger.info(f"Removed {initial_count - final_count} duplicate records")

# 4. Write as a single parquet file
logger.info("Starting to write parquet file...")
transformed_data.coalesce(1).write.mode("overwrite").parquet(output_path)
logger.info("Parquet file write completed")

# Get final count
final_count = transformed_data.count()
logger.info(f"Total records processed: {final_count}")

# Log job completion
logger.info(f"Job completed successfully")

# Commit the job
job.commit()