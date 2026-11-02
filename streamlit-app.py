import time
import numpy as np
import pandas as pd
import streamlit as st
import psycopg2
import os
import plotly.express as px
from pathlib import Path
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "dbname": os.getenv("POSTGRES_DB", "fraud_detection"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "port": int(os.getenv("POSTGRES_PORT", "5432"))
}


RULE_ALERT_THRESHOLD = 1500.0

st.set_page_config(page_title="Real-Time Fraud Dashboard", layout="wide")
st.title("Real-Time Financial Fraud Detection Dashboard")

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def initialize_tables():
    conn = get_connection()
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
        CREATE INDEX IF NOT EXISTS idx_prediction_results_time
        ON prediction_results(processed_at)
    """)

    conn.commit()
    cur.close()
    conn.close()

# Cache queries to avoid hitting PostgreSQL on every dashboard interaction
@st.cache_data(ttl=5)
def fetch_stats():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            COUNT(*),
            COUNT(DISTINCT customer_id),
            COUNT(DISTINCT merchant_id),
            COUNT(*) FILTER (WHERE predicted_fraud = 1),
            COUNT(*) FILTER (WHERE amount >= %s),
            COALESCE(SUM(amount), 0)
        FROM prediction_results
    """, (RULE_ALERT_THRESHOLD,))

    result = cur.fetchone()
    cur.close()
    conn.close()

    return result

@st.cache_data(ttl=5)
def fetch_data(query, params=None):
    conn = get_connection()

    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

def split_frame(input_df, rows):
    return [
        input_df.iloc[i:i + rows]
        for i in range(0, len(input_df), rows)
    ]

def paginate_table(table_data, key):
    if table_data.empty:
        return

    top_menu = st.columns(3)

    with top_menu[0]:
        sort = st.radio(
            "Sort Data",
            options=["Yes", "No"],
            horizontal=True,
            index=1,
            key=f"{key}_sort"
        )

    if sort == "Yes":
        with top_menu[1]:
            sort_field = st.selectbox(
                "Sort By",
                options=table_data.columns,
                key=f"{key}_field"
            )

        with top_menu[2]:
            sort_direction = st.radio(
                "Direction",
                options=["⬆️", "⬇️"],
                horizontal=True,
                key=f"{key}_direction"
            )

        table_data = table_data.sort_values(
            by=sort_field,
            ascending=sort_direction == "⬆️",
            ignore_index=True
        )

    bottom_menu = st.columns((4, 1, 1))

    with bottom_menu[2]:
        batch_size = st.selectbox(
            "Page Size",
            options=[10, 25, 50, 100],
            key=f"{key}_size"
        )

    total_pages = max(
        1,
        int(np.ceil(len(table_data) / batch_size))
    )

    page_key = f"{key}_page"

    if st.session_state.get(page_key, 1) > total_pages:
        st.session_state[page_key] = total_pages

    with bottom_menu[1]:
        current_page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            step=1,
            key=page_key
        )

    with bottom_menu[0]:
        st.markdown(f"Page **{current_page}** of **{total_pages}**")

    pages = split_frame(table_data, batch_size)

    if pages:
        st.dataframe(
            pages[current_page - 1],
            use_container_width=True,
            hide_index=True
        )

def plot_fraud_distribution():
    fraud_df = fetch_data("""
        SELECT
            CASE
                WHEN predicted_fraud = 1 THEN 'Flagged as Suspicious'
                ELSE 'Classified as Legitimate'
            END AS prediction,
            COUNT(*) AS total_transactions
        FROM prediction_results
        GROUP BY predicted_fraud
    """)

    if fraud_df.empty:
        st.info("Waiting for transaction predictions...")
        return

    fig = px.pie(
        fraud_df,
        names="prediction",
        values="total_transactions",
        title="Transaction Classification Distribution",
        color="prediction",
        color_discrete_map={
            "Classified as Legitimate": "#16A34A",
            "Flagged as Suspicious": "#DC5555"
        },
        hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_revenue(revenue_df):
    top_merchants = revenue_df.sort_values(
        by="total_revenue",
        ascending=False
    ).head(10)

    fig = px.bar(
        top_merchants.sort_values("total_revenue"),
        x="total_revenue",
        y="merchant_id",
        orientation="h",
        labels={
            "total_revenue": "Transaction Amount",
            "merchant_id": "Merchant"
        },
        title="Top 10 Merchants by Transaction Amount"
    )

    st.plotly_chart(fig, use_container_width=True)

# Read the historical and live predictions together to maintain for one dashboard
def update_data():
    stats = fetch_stats()

    total_transactions = stats[0]
    customers_count = stats[1]
    merchants_count = stats[2]
    ml_alert_count = stats[3]
    rule_alert_count = stats[4]
    total_amount = float(stats[5])

    st.caption(f"Last refresh: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    st.header("Transaction Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Registered Customers",
        f"{customers_count:,}"
    )

    col3.metric(
        "Active Merchants",
        f"{merchants_count:,}"
    )

    col4.metric(
        "Total Transaction Amount",
        f"${total_amount:,.2f}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "ML Fraud Alerts",
        f"{ml_alert_count:,}"
    )

    col2.metric(
        "High-Value Transactions",
        f"{rule_alert_count:,}"
    )

    col3.metric(
        "Fraud Detection Rate",
        f"{ml_alert_count / total_transactions:.2%}"
        if total_transactions else "0.00%"
    )

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        plot_fraud_distribution()

    revenue_df = fetch_data("""
        SELECT
            merchant_id,
            COUNT(*) AS total_transactions,
            SUM(amount)::DOUBLE PRECISION AS total_revenue
        FROM prediction_results
        GROUP BY merchant_id
        ORDER BY total_revenue DESC
    """)

    with col2:
        if not revenue_df.empty:
            plot_revenue(revenue_df)

    if not revenue_df.empty:
        st.subheader("Revenue Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Transaction Amount",
            f"${revenue_df['total_revenue'].sum():,.2f}"
        )

        col2.metric(
            "Active Merchants",
            len(revenue_df)
        )

        col3.metric(
            "Average Amount per Merchant",
            f"${revenue_df['total_revenue'].mean():,.2f}"
        )

    st.markdown("---")
    st.header("Rule-Based High-Value Alerts")

    rule_alerts = fetch_data("""
        SELECT
            transaction_time,
            transaction_id,
            customer_id,
            merchant_id,
            category,
            amount,
            fraud_score,
            predicted_fraud
        FROM prediction_results
        WHERE amount >= %s
        ORDER BY transaction_time DESC
    """, (RULE_ALERT_THRESHOLD,))

    if not rule_alerts.empty:
        for _, row in rule_alerts.head(3).iterrows():
            st.warning(
                f"Transaction {row['transaction_id']} | "
                f"Customer {row['customer_id']} | "
                f"Amount: ${row['amount']:,.2f}"
            )

        st.subheader("Complete High-Value Transaction Log")
        paginate_table(rule_alerts, "rule_alerts")
    else:
        st.info("No high-value transactions found.")

    st.markdown("---")
    st.header("Machine Learning Fraud Detection")

    ml_alerts = fetch_data("""
        SELECT
            transaction_time,
            transaction_id,
            customer_id,
            merchant_id,
            category,
            amount,
            fraud_score
        FROM prediction_results
        WHERE predicted_fraud = 1
        ORDER BY fraud_score DESC, processed_at DESC
    """)

    if not ml_alerts.empty:
        col1, col2 = st.columns(2)

        col1.metric(
            "ML-Flagged Transactions",
            f"{len(ml_alerts):,}"
        )

        col2.metric(
            "Average Fraud Score",
            f"{ml_alerts['fraud_score'].mean():.2%}"
        )

        st.subheader("Highest-Risk Transactions")

        for _, row in ml_alerts.head(3).iterrows():
            st.error(
                f"Transaction: {row['transaction_id']} | "
                f"Customer: {row['customer_id']} | "
                f"Amount: ${row['amount']:,.2f} | "
                f"Fraud Score: {row['fraud_score']:.2%}"
            )

        st.subheader("Complete ML Fraud Detection Log")
        paginate_table(ml_alerts, "ml_alerts")
    else:
        st.success("No suspicious transactions detected.")

    st.markdown("---")
    st.header("Complete Transaction History")

    all_transactions = fetch_data("""
        SELECT
            transaction_time,
            transaction_id,
            customer_id,
            merchant_id,
            category,
            amount,
            fraud_score,
            predicted_fraud,
            CASE
                WHEN actual_fraud = 1 THEN 'Fraudulent'
                WHEN actual_fraud = 0 THEN 'Legitimate'
                ELSE 'Not Available'
            END AS actual_status,
            CASE
                WHEN predicted_fraud = 1 THEN 'Suspicious'
                ELSE 'Legitimate'
            END AS predicted_status
        FROM prediction_results
        ORDER BY processed_at DESC, transaction_id DESC
    """)

    if not all_transactions.empty:
        paginate_table(
            all_transactions,
            "all_transactions"
        )

    else:
        st.info("Waiting for transaction predictions...")

def main():
    try:
        initialize_tables()

        refresh_interval = st.sidebar.slider(
            "Refresh interval (seconds)",
            5, 60, 10
        )

        if st.sidebar.button("Refresh Now"):
            st.cache_data.clear()
            st.rerun()

        st_autorefresh(
            interval=refresh_interval * 1000,
            key="auto"
        )

        update_data()

    except psycopg2.Error as e:
        st.error(f"PostgreSQL connection error: {e}")
        st.info("Check that PostgreSQL is running and the fraud_detection database exists.")

if __name__ == "__main__":
    main()
