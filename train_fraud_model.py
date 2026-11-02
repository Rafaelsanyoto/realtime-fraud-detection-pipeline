from pathlib import Path
from math import sqrt
import psycopg2
import os
from psycopg2.extras import execute_values
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, sum as _sum
from pyspark.ml import Pipeline
from dotenv import load_dotenv
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

BASE_DIR = Path(__file__).resolve().parent
JDBC_JAR = BASE_DIR / "postgresql-42.7.14.jar"
MODEL_PATH = BASE_DIR / "models" / "fraud_model"
load_dotenv(BASE_DIR / ".env")

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "dbname": os.getenv("POSTGRES_DB", "fraud_detection"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "port": int(os.getenv("POSTGRES_PORT", "5432"))
}
TARGET_RECALL = 0.80
THRESHOLDS = [round(i / 100, 2) for i in range(1, 100)]

def save_historical_predictions(rows):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
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
            DELETE FROM prediction_results
            WHERE source_type = 'historical'
        """)

        execute_values(cur, """
            INSERT INTO prediction_results (
                transaction_id, customer_id, merchant_id, category,
                amount, transaction_time, step, actual_fraud,
                fraud_score, predicted_fraud, source_type, data_split
            ) VALUES %s
            ON CONFLICT (transaction_id) DO NOTHING
        """, rows)

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        cur.close()
        conn.close()

def main():
    if not JDBC_JAR.exists():
        raise FileNotFoundError(f"JDBC driver not found: {JDBC_JAR}")

    spark = (SparkSession.builder
        .appName("FraudModelTraining")
        .config("spark.jars", str(JDBC_JAR))
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    try:
        jdbc_url = (
            f"jdbc:postgresql://{DB_CONFIG['host']}:"
            f"{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
        )

        properties = {
            "user": DB_CONFIG["user"],
            "password": DB_CONFIG["password"],
            "driver": "org.postgresql.Driver"
        }

        query = """
        (
            SELECT
                t.transaction_id,
                t.customer_id,
                t.merchant_id,
                t.amount,
                t.transaction_time,
                t.step,
                t.is_fraud,
                t.data_split,
                m.category
            FROM transactions t
            JOIN merchants m ON t.merchant_id = m.merchant_id
        ) AS training_data
        """

        df = spark.read.jdbc(
            url=jdbc_url,
            table=query,
            properties=properties
        )

        df = (df
            .withColumn("amount", col("amount").cast("double"))
            .withColumn("step", col("step").cast("double"))
            .withColumn("label", col("is_fraud").cast("double"))
            .dropna(subset=["amount", "step", "category", "label"])
        )

        train_df = df.filter(col("data_split") == "train")
        test_df = df.filter(col("data_split") == "test")

        fit_df, validation_df = train_df.randomSplit([0.85, 0.15], seed=42)

        fit_counts = {
            int(row["label"]): row["count"]
            for row in fit_df.groupBy("label").count().collect()
        }

        validation_counts = {
            int(row["label"]): row["count"]
            for row in validation_df.groupBy("label").count().collect()
        }

        test_counts = {
            int(row["label"]): row["count"]
            for row in test_df.groupBy("label").count().collect()
        }

        fraud_count = fit_counts.get(1, 0)
        legitimate_count = fit_counts.get(0, 0)

        if fraud_count == 0 or legitimate_count == 0:
            raise ValueError("Training data must contain both fraud classes.")

        if validation_counts.get(1, 0) == 0 or validation_counts.get(0, 0) == 0:
            raise ValueError("Validation data must contain both fraud classes.")

        if test_counts.get(1, 0) == 0 or test_counts.get(0, 0) == 0:
            raise ValueError("Testing data must contain both fraud classes.")

        print("Training distribution:", fit_counts)
        print("Validation distribution:", validation_counts)
        print("Testing distribution:", test_counts)

        fraud_weight = sqrt(legitimate_count / fraud_count)

        fit_df = fit_df.withColumn(
            "class_weight",
            when(col("label") == 1, lit(fraud_weight)).otherwise(lit(1.0))
        )

        # Convert the merchant categories into numerical features for the classifier
        category_indexer = StringIndexer(
            inputCol="category",
            outputCol="category_index",
            handleInvalid="keep"
        )

        category_encoder = OneHotEncoder(
            inputCols=["category_index"],
            outputCols=["category_vector"],
            handleInvalid="keep"
        )

        assembler = VectorAssembler(
            inputCols=["amount", "step", "category_vector"],
            outputCol="features"
        )

        rf = RandomForestClassifier(
            labelCol="label",
            featuresCol="features",
            weightCol="class_weight",
            numTrees=100,
            maxDepth=8,
            maxBins=32,
            seed=42
        )

        pipeline = Pipeline(stages=[
            category_indexer,
            category_encoder,
            assembler,
            rf
        ])

        model = pipeline.fit(fit_df)
        validation_predictions = model.transform(validation_df)

        # Use validation predictions to select a threshold targeting at least an 80% recall
        validation_scores = validation_predictions.select(
            col("label"),
            vector_to_array(col("probability"))[1].alias("fraud_score")
        ).collect()

        threshold_results = []

        for threshold in THRESHOLDS:
            tp = sum(1 for row in validation_scores if row["label"] == 1 and row["fraud_score"] >= threshold)
            fp = sum(1 for row in validation_scores if row["label"] == 0 and row["fraud_score"] >= threshold)
            fn = sum(1 for row in validation_scores if row["label"] == 1 and row["fraud_score"] < threshold)

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            threshold_results.append({
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

        eligible = [
            result for result in threshold_results
            if result["recall"] >= TARGET_RECALL
        ]

        if eligible:
            best_result = max(
                eligible,
                key=lambda x: (x["precision"], x["f1"], x["threshold"])
            )
        else:
            best_result = max(
                threshold_results,
                key=lambda x: (x["f1"], x["recall"], x["threshold"])
            )

        best_threshold = best_result["threshold"]

        print("\nSelected threshold:", best_threshold)
        print("Validation precision:", best_result["precision"])
        print("Validation recall:", best_result["recall"])
        print("Validation F1-score:", best_result["f1"])

        model.stages[-1].setThresholds([
            1.0 - best_threshold,
            best_threshold
        ])

        test_predictions = model.transform(test_df).cache()

        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderPR"
        )

        pr_auc = evaluator.evaluate(test_predictions)

        counts = test_predictions.agg(
            _sum(when((col("label") == 1) & (col("prediction") == 1), 1).otherwise(0)).alias("tp"),
            _sum(when((col("label") == 0) & (col("prediction") == 1), 1).otherwise(0)).alias("fp"),
            _sum(when((col("label") == 1) & (col("prediction") == 0), 1).otherwise(0)).alias("fn"),
            _sum(when((col("label") == 0) & (col("prediction") == 0), 1).otherwise(0)).alias("tn")
        ).first()

        tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

        print("\nFinal Test Results")
        print("PR-AUC:", pr_auc)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)
        print("Accuracy:", accuracy)
        print("True Positives:", tp)
        print("False Positives:", fp)
        print("False Negatives:", fn)
        print("True Negatives:", tn)

        test_predictions.groupBy("label", "prediction").count().show()
        test_predictions.unpersist()

        historical_scored = model.transform(df)

        historical_rows = [
            (
                row.transaction_id,
                row.customer_id,
                row.merchant_id,
                row.category,
                float(row.amount),
                row.transaction_time,
                int(row.step),
                int(row.actual_fraud),
                float(row.fraud_score),
                int(row.predicted_fraud),
                "historical",
                row.data_split
            )
            for row in historical_scored.select(
                "transaction_id",
                "customer_id",
                "merchant_id",
                "category",
                "amount",
                "transaction_time",
                "step",
                col("label").cast("int").alias("actual_fraud"),
                vector_to_array(col("probability"))[1].alias("fraud_score"),
                col("prediction").cast("int").alias("predicted_fraud"),
                "data_split"
            ).collect()
        ]

        save_historical_predictions(historical_rows)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.write().overwrite().save(str(MODEL_PATH))

        print(f"Historical predictions saved: {len(historical_rows)}")
        print(f"Model saved to {MODEL_PATH}")

    finally:
        spark.stop()

if __name__ == "__main__":
    main()
