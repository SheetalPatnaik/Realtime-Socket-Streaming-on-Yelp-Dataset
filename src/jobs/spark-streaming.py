from time import sleep
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, when, udf
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import openai
from config.config import config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Function for sentiment analysis
def sentiment_analysis(comment: str) -> str:
    if comment:
        try:
            # Initialize OpenAI API key
            openai.api_key = config['openai']['api_key']

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"""
                            You're a machine learning model with a task of classifying comments into POSITIVE, NEGATIVE, or NEUTRAL.
                            You are to respond with one word from the options specified above. Do not add anything else.
                            Here is the comment:
                            
                            {comment}
                        """
                    }
                ]
            )

            # Return classification
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            # Log the exact exception for debugging
            print(f"Error during sentiment analysis for comment: {comment}\nException: {e}")
            return "ERROR"
    return "Empty"


# Main streaming function
def start_streaming(spark):
    topic = 'customers_review'

    while True:
        try:
            # Read data from the socket
            stream_df = (spark.readStream.format("socket")
                         .option("host", "0.0.0.0")
                         .option("port", 9999)
                         .load()
                         )

            # Define schema for the incoming data
            schema = StructType([
                StructField("review_id", StringType()),
                StructField("user_id", StringType()),
                StructField("business_id", StringType()),
                StructField("stars", FloatType()),
                StructField("date", StringType()),
                StructField("text", StringType())
            ])

            # Parse JSON data and extract fields
            stream_df = stream_df.select(from_json(col('value'), schema).alias("data")).select("data.*")

            # Register the UDF for sentiment analysis
            sentiment_analysis_udf = udf(sentiment_analysis, StringType())

            # Apply sentiment analysis to the 'text' column
            stream_df = stream_df.withColumn(
                'feedback',
                when(col('text').isNotNull(), sentiment_analysis_udf(col('text')))
                .otherwise("Empty")
            )

            # Prepare data for writing to Kafka
            kafka_df = stream_df.selectExpr("CAST(review_id AS STRING) AS key", "to_json(struct(*)) AS value")

            # Write stream back to Kafka
            checkpoint_path = f"/tmp/checkpoint-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            query = (kafka_df.writeStream
                     .format("kafka")
                     .option("kafka.bootstrap.servers", config['kafka']['bootstrap.servers'])
                     .option("kafka.security.protocol", config['kafka']['security.protocol'])
                     .option("kafka.sasl.mechanism", config['kafka']['sasl.mechanisms'])
                     .option("kafka.sasl.jaas.config",
                             f'org.apache.kafka.common.security.plain.PlainLoginModule required username="{config["kafka"]["sasl.username"]}" '
                             f'password="{config["kafka"]["sasl.password"]}";')
                     .option("checkpointLocation", checkpoint_path)
                     .option("topic", topic)
                     .start()
                     .awaitTermination()
                     )

        except Exception as e:
            logging.error(f"Exception encountered: {e}. Retrying in 10 seconds")
            sleep(10)

# Entry point
if __name__ == "__main__":
    spark_conn = SparkSession.builder \
        .appName("SocketStreamConsumer") \
        .getOrCreate()

    start_streaming(spark_conn)
