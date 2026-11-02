import csv
import random
import os
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "dbname": os.getenv("POSTGRES_DB", "fraud_detection"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "port": int(os.getenv("POSTGRES_PORT", "5432"))
}
DATASET_PATH = BASE_DIR / "bs140513_032310.csv"
SIMULATION_START = datetime(2025, 1, 1)
HISTORICAL_LIMIT = 10000
TRAIN_RATIO = 0.8
BATCH_SIZE = 2000

def clean(value):
    return value.strip().strip('"').strip("'") if value else ""


def read_historical_sample():
    counts = {0:0,1:0}

    with open(DATASET_PATH, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"step", "customer", "age", "gender", "zipcodeOri", "merchant", "zipMerchant", "category", "amount", "fraud"}

        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Unexpected BankSim CSV columns.")

        for row in reader:
            label = int(clean(row["fraud"]))
            if label not in counts:
                raise ValueError(f"Unexpected fraud label: {label}")
            counts[label] += 1

    total = sum(counts.values())

    if total < HISTORICAL_LIMIT or not counts[0] or not counts[1]:
        raise ValueError("Dataset does not contain enough records from both classes.")

    fraud_target = round(HISTORICAL_LIMIT * counts[1] / total)
    targets = {
        0: HISTORICAL_LIMIT - fraud_target,
        1: fraud_target
    }

    reservoirs = {0: [], 1: []}
    seen = {0: 0, 1: 0}
    rng = random.Random(42)

    with open(DATASET_PATH, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(reader, start=1):
            label = int(clean(row["fraud"]))
            seen[label] += 1

            record = {
                "customer_id": clean(row["customer"]),
                "merchant_id": clean(row["merchant"]),
                "age_group": clean(row["age"]),
                "gender": clean(row["gender"]),
                "zipcode_ori": clean(row["zipcodeOri"]),
                "zip_merchant": clean(row["zipMerchant"]),
                "category": clean(row["category"]),
                "step": int(clean(row["step"])),
                "amount": float(clean(row["amount"])),
                "is_fraud": label
            }

            if len(reservoirs[label]) < targets[label]:
                reservoirs[label].append((row_number, record))
            else:
                position = rng.randrange(seen[label])
                if position < targets[label]:
                    reservoirs[label][position] = (row_number, record)

    selected = []

    for label in (0, 1):
        group = reservoirs[label]
        rng.shuffle(group)
        train_count = int(len(group) * TRAIN_RATIO)

        if len(group) > 1:
            train_count = max(1, min(train_count, len(group) - 1))

        for index, item in enumerate(group):
            row_number, record = item
            data_split = "train" if index < train_count else "test"
            selected.append((row_number, record, data_split))

    rng.shuffle(selected)
    return selected

def create_tables(cur):
    cur.execute("DROP TABLE IF EXISTS prediction_results CASCADE")
    cur.execute("DROP TABLE IF EXISTS streaming_predictions CASCADE")
    cur.execute("DROP TABLE IF EXISTS transactions CASCADE")
    cur.execute("DROP TABLE IF EXISTS customers CASCADE")
    cur.execute("DROP TABLE IF EXISTS merchants CASCADE")

    cur.execute("""
        CREATE TABLE merchants(
            merchant_id VARCHAR(255) PRIMARY KEY,
            category VARCHAR(255),
            zip_merchant VARCHAR(50)
        )
    """)

    cur.execute("""
        CREATE TABLE customers(
            customer_id VARCHAR(255) PRIMARY KEY,
            age_group VARCHAR(20),
            gender VARCHAR(20),
            zipcode_ori VARCHAR(50)
        )
    """)

    cur.execute("""
        CREATE TABLE transactions(
            transaction_id VARCHAR(255) PRIMARY KEY,
            customer_id VARCHAR(255) REFERENCES customers(customer_id),
            merchant_id VARCHAR(255) REFERENCES merchants(merchant_id),
            amount NUMERIC(12, 2),
            transaction_time TIMESTAMP,
            step INTEGER,
            is_fraud SMALLINT CHECK (is_fraud IN (0, 1)),
            data_split VARCHAR(10) CHECK (data_split IN ('train', 'test'))
        )
    """)

    cur.execute("CREATE INDEX idx_transactions_split ON transactions(data_split, step)")
    cur.execute("CREATE INDEX idx_transactions_customer ON transactions(customer_id)")
    cur.execute("CREATE INDEX idx_transactions_merchant ON transactions(merchant_id)")

def save_batch(conn, cur, customers, merchants, transactions):
    if customers:
        execute_values(cur, """
            INSERT INTO customers(customer_id, age_group, gender, zipcode_ori)
            VALUES %s ON CONFLICT (customer_id) DO NOTHING
        """, list(customers.values()))

    if merchants:
        execute_values(cur, """
            INSERT INTO merchants(merchant_id, category, zip_merchant)
            VALUES %s ON CONFLICT (merchant_id) DO NOTHING
        """, list(merchants.values()))

    if transactions:
        execute_values(cur, """
            INSERT INTO transactions(
                transaction_id, customer_id, merchant_id, amount,
                transaction_time, step, is_fraud, data_split
            ) VALUES %s ON CONFLICT (transaction_id) DO NOTHING
        """, transactions)

    conn.commit()

def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    selected = read_historical_sample()
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        create_tables(cur)
        conn.commit()

        customer_batch = {}
        merchant_batch = {}
        transaction_batch = []

        rows_loaded = 0

        for source_row_number, record, data_split in selected:
            rows_loaded += 1
            customer_id = record["customer_id"]
            merchant_id = record["merchant_id"]
            step = record["step"]

            customer_batch[customer_id] = (
                customer_id,
                record["age_group"],
                record["gender"],
                record["zipcode_ori"]
            )

            merchant_batch[merchant_id] = (
                merchant_id,
                record["category"],
                record["zip_merchant"]
            )

            transaction_batch.append((
                f"BS{source_row_number:08d}",
                customer_id,
                merchant_id,
                record["amount"],
                SIMULATION_START + timedelta(days=step),
                step,
                record["is_fraud"],
                data_split
            ))

            if len(transaction_batch) >= BATCH_SIZE:
                save_batch(conn, cur, customer_batch, merchant_batch, transaction_batch)
                customer_batch.clear()
                merchant_batch.clear()
                transaction_batch.clear()
                print(f"Loaded {rows_loaded} transactions")

        save_batch(conn, cur, customer_batch, merchant_batch, transaction_batch)

        cur.execute("""
            SELECT data_split, is_fraud, COUNT(*)
            FROM transactions
            GROUP BY data_split, is_fraud
            ORDER BY data_split, is_fraud
        """)

        for result in cur.fetchall():
            print(result)

        cur.execute("SELECT COUNT(*) FROM transactions")
        print(f"Total historical transactions: {cur.fetchone()[0]}")

    except Exception:
        conn.rollback()
        raise

    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()
