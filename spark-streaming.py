from pathlib import Path
from uuid import uuid4
import os
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, sum as _sum, lit, struct, to_json
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, IntegerType
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "dbname": os.getenv("POSTGRES_DB", "fraud_detection"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "port": int(os.getenv("POSTGRES_PORT", "5432"))
}

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")

if not KAFKA_BOOTSTRAP_SERVERS:
    raise ValueError("KAFKA_BOOTSTRAP_SERVERS is missing from .env")

MODEL_PATH = BASE_DIR / "models" / "fraud_model"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RUN_ID = str(uuid4())

# The historical and live predictions share one table for dashboard monitoring
def create_prediction_table():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_results (
            transaction_id VARCHAR(255) PRIMARY KEY,
            customer_id VARCHAR(255),
            merchant_id VARCHAR(255),
            category VARCHAR(255),
            amount NUMERIC(12, 2),
            transaction_time TIMESTAMP,
            step INTEGER,
            actual_fraud SMALLINT,
            fraud_score DOUBLE PRECISION,
            predicted_fraud SMALLINT,
            source_type VARCHAR(20),
            data_split VARCHAR(10),
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_prediction_results_source
        ON prediction_results(source_type)
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_prediction_results_time
        ON prediction_results(processed_at)
    """)

    conn.commit()
    cur.close()
    conn.close()

def publish_to_kafka(df, topic, key_column):
    if not df.take(1):
        return

    messages = df.select(
        col(key_column).cast("string").alias("key"),
        to_json(struct(*[col(name) for name in df.columns])).alias("value")
    )

    messages.write \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("topic", topic) \
        .save()

def publish_revenue(batch_df, batch_id):
    if not batch_df.take(1):
        return

    revenue_df = batch_df.groupBy("merchant_id").agg(
        _sum("amount").alias("total_revenue")
    ).withColumn(
        "batch_id", lit(batch_id)
    ).withColumn(
        "run_id", lit(RUN_ID)
    )

    publish_to_kafka(
        revenue_df,
        "banksim_merchant_revenue",
        "merchant_id"
    )

def save_streaming_predictions(scored_df):
    rows = [
        (
            row.transaction_id,
            row.customer_id,
            row.merchant_id,
            row.category,
            float(row.amount),
            row.transaction_time,
            int(row.step),
            None,
            float(row.fraud_score),
            int(row.predicted_fraud),
            "live",
            None
        )
        for row in scored_df.select(
            "transaction_id",
            "customer_id",
            "merchant_id",
            "category",
            "amount",
            "transaction_time",
            "step",
            "fraud_score",
            "predicted_fraud"
        ).collect()
    ]

    if not rows:
        return

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        # Avoid inserting the same transaction twice if Spark retries a micro-batch
        execute_values(cur, """
            INSERT INTO prediction_results (
                transaction_id, customer_id, merchant_id, category,
                amount, transaction_time, step, actual_fraud,
                fraud_score, predicted_fraud, source_type, data_split
            ) VALUES %s
            ON CONFLICT (transaction_id) DO NOTHING
        """, rows)

        conn.commit()
        print(f"Saved {len(rows)} live predictions to PostgreSQL")

    except Exception:
        conn.rollback()
        raise

    finally:
        cur.close()
        conn.close()

def score_transactions(batch_df, batch_id, fraud_model):
    if not batch_df.take(1):
        return

    feature_df = batch_df.withColumn(
        "step",
        col("step").cast("double")
    )

    predictions = fraud_model.transform(feature_df)

    scored_df = predictions.select(
        "transaction_id",
        "customer_id",
        "merchant_id",
        "category",
        "amount",
        "transaction_time",
        col("step").cast("int").alias("step"),
        vector_to_array(col("probability"))[1].alias("fraud_score"),
        col("prediction").cast("int").alias("predicted_fraud")
    ).cache()

    try:
        save_streaming_predictions(scored_df)

        publish_to_kafka(
            scored_df,
            "banksim_scored_transactions",
            "transaction_id"
        )

        ml_alerts = scored_df.filter(
            col("predicted_fraud") == 1
        )

        publish_to_kafka(
            ml_alerts,
            "banksim_ml_fraud_alerts",
            "transaction_id"
        )

        print(f"Processed ML batch {batch_id}")

    finally:
        scored_df.unpersist()

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    spark = (SparkSession.builder
        .appName("FinancialFraudDetectionEngineering")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0"
        )
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    fraud_model = PipelineModel.load(str(MODEL_PATH))
    create_prediction_table()

    transaction_schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("merchant_id", StringType(), True),
        StructField("category", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("transaction_time", TimestampType(), True),
        StructField("step", IntegerType(), True)
    ])

    kafka_df = (spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", "banksim_transactions")
        .option("startingOffsets", "earliest")
        .load()
    )

    transactions_df = (kafka_df
        .selectExpr("CAST(value AS STRING) AS value")
        .select(from_json(col("value"), transaction_schema).alias("data"))
        .select("data.*")
        .filter(col("transaction_id").isNotNull())
        .filter(col("amount").isNotNull())
        .filter(col("step").isNotNull())
        .filter(col("category").isNotNull())
    )

    flagged_high_value = transactions_df.filter(
        col("amount") >= 1500.0
    )

    rule_alert_query = (flagged_high_value
        .select(
            col("transaction_id").cast("string").alias("key"),
            to_json(struct(*[col(name) for name in flagged_high_value.columns])).alias("value")
        )
        .writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("topic", "banksim_flagged_alerts")
        .option("checkpointLocation", str(CHECKPOINT_DIR / "rule_alerts"))
        .outputMode("append")
        .start()
    )

    revenue_query = (transactions_df.writeStream
        .foreachBatch(publish_revenue)
        .option("checkpointLocation", str(CHECKPOINT_DIR / "revenue"))
        .start()
    )

    ml_query = (transactions_df.writeStream
        .foreachBatch(
            lambda batch_df, batch_id: score_transactions(
                batch_df, batch_id, fraud_model
            )
        )
        .option("checkpointLocation", str(CHECKPOINT_DIR / "ml_predictions"))
        .start()
    )

    spark.streams.awaitAnyTermination()
