from pathlib import Path
import sqlite3
import pandas as pd

UNIFIED_COLUMNS = [
    "entity_id", "event_id", "customer_id", "timestamp",
    "amount", "is_cancelled", "category", "source",
]


def _require_files(base, filenames):
    missing = [f for f in filenames if not (base / f).exists()]
    if missing:
        raise FileNotFoundError(f"В {base} не найдены файлы: {missing}")


def load_olist(base_dir="data/olist"):
    base = Path(base_dir)
    _require_files(base, [
        "olist_orders_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_order_payments_dataset.csv",
    ])

    orders = pd.read_csv(base / "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
    items = pd.read_csv(base / "olist_order_items_dataset.csv")
    payments = pd.read_csv(base / "olist_order_payments_dataset.csv")

    df = items.merge(orders[["order_id", "customer_id", "order_purchase_timestamp", "order_status"]], on="order_id", how="left")
    payments_agg = payments.groupby("order_id", as_index=False)["payment_value"].sum()
    df = df.merge(payments_agg, on="order_id", how="left")

    unified = pd.DataFrame({
        "entity_id": df["seller_id"],
        "event_id": df["order_id"] + "_" + df["order_item_id"].astype(str),
        "customer_id": df["customer_id"],
        "timestamp": df["order_purchase_timestamp"],
        "amount": df["price"],
        "is_cancelled": df["order_status"].isin(["canceled", "unavailable"]),
        "category": pd.NA,
        "source": "olist_brazil",
    })
    return unified


def load_olist_from_sqlite(db_path="olist_db.db"):
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Файл базы {db_path} не найден")

    conn = sqlite3.connect(db_path)
    try:
        query = """
        SELECT
            oi.seller_id AS entity_id,
            oi.order_id || '_' || oi.order_item_id AS event_id,
            o.customer_id AS customer_id,
            o.order_purchase_timestamp AS timestamp,
            oi.price AS amount,
            o.order_status AS order_status,
            COALESCE(p.total_payment, 0) AS total_payment
        FROM olist_order_items_dataset oi
        LEFT JOIN olist_orders_dataset o ON oi.order_id = o.order_id
        LEFT JOIN (
            SELECT order_id, SUM(payment_value) AS total_payment
            FROM olist_order_payments_dataset
            GROUP BY order_id
        ) p ON oi.order_id = p.order_id
        """
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    unified = pd.DataFrame({
        "entity_id": df["entity_id"],
        "event_id": df["event_id"],
        "customer_id": df["customer_id"],
        "timestamp": df["timestamp"],
        "amount": df["amount"],
        "is_cancelled": df["order_status"].isin(["canceled", "unavailable"]),
        "category": pd.NA,
        "source": "olist_brazil",
    })
    return unified


def load_fraud_ecommerce(base_dir="data/fraud_ecommerce"):
    base = Path(base_dir)
    _require_files(base, ["Fraud_Data.csv"])
    df = pd.read_csv(base / "Fraud_Data.csv", parse_dates=["signup_time", "purchase_time"])
    unified = pd.DataFrame({
        "entity_id": df["device_id"],
        "event_id": df["user_id"].astype(str) + "_" + df["purchase_time"].astype(str),
        "customer_id": df["user_id"],
        "timestamp": df["purchase_time"],
        "amount": df["purchase_value"],
        "is_cancelled": df["class"] == 1,
        "category": df.get("source", pd.NA),
        "source": "fraud_ecommerce_us",
    })
    return unified


def load_uci_retail(base_dir="data/uci_retail"):
    base = Path(base_dir)
    filename = "OnlineRetail.csv"
    _require_files(base, [filename])
    df = pd.read_csv(base / filename, encoding="ISO-8859-1", parse_dates=["InvoiceDate"])
    df["is_cancelled"] = df["InvoiceNo"].astype(str).str.startswith("C")
    unified = pd.DataFrame({
        "entity_id": df["CustomerID"],
        "event_id": df["InvoiceNo"].astype(str) + "_" + df["StockCode"].astype(str),
        "customer_id": df["CustomerID"],
        "timestamp": df["InvoiceDate"],
        "amount": df["Quantity"] * df["UnitPrice"],
        "is_cancelled": df["is_cancelled"],
        "category": df["Description"],
        "source": "uk_online_retail",
    })
    return unified


def load_instacart(base_dir="data/instacart"):
    base = Path(base_dir)
    _require_files(base, ["orders.csv"])
    orders = pd.read_csv(base / "orders.csv")
    unified = pd.DataFrame({
        "entity_id": orders["user_id"],
        "event_id": orders["order_id"].astype(str),
        "customer_id": orders["user_id"],
        "timestamp": pd.NA,
        "amount": pd.NA,
        "is_cancelled": False,
        "category": pd.NA,
        "source": "instacart_grocery",
    })
    return unified