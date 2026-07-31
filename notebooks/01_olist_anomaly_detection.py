import math
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

project_root = Path(__file__).resolve().parent.parent
db_path = project_root / "olist_db.db"
report_dir = project_root / "report"
report_dir.mkdir(exist_ok=True)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

print("Считаем фичи по продавцам ")
print()

seller_features_query = """
WITH order_seller AS (
    SELECT
        oi.seller_id,
        oi.order_id,
        o.customer_id,
        o.order_status,
        oi.price
    FROM olist_order_items_dataset oi
    JOIN olist_orders_dataset o ON oi.order_id = o.order_id
),
base AS (
    SELECT
        seller_id,
        COUNT(*) AS n_events,
        COUNT(DISTINCT customer_id) AS n_unique_customers,
        AVG(CASE WHEN order_status IN ('canceled', 'unavailable') THEN 1.0 ELSE 0.0 END) AS cancellation_rate,
        AVG(price) AS avg_amount,
        SQRT(AVG(price * price) - AVG(price) * AVG(price)) AS amount_std
    FROM order_seller
    GROUP BY seller_id
),
customer_counts AS (
    SELECT seller_id, customer_id, COUNT(*) AS cnt
    FROM order_seller
    GROUP BY seller_id, customer_id
),
ranked AS (
    SELECT
        seller_id,
        cnt,
        ROW_NUMBER() OVER (PARTITION BY seller_id ORDER BY cnt DESC) AS rn,
        SUM(cnt) OVER (PARTITION BY seller_id) AS total_cnt
    FROM customer_counts
),
concentration AS (
    SELECT seller_id, SUM(cnt) * 1.0 / MAX(total_cnt) AS top3_customer_concentration
    FROM ranked
    WHERE rn <= 3
    GROUP BY seller_id
)
SELECT
    base.seller_id,
    base.n_events,
    base.n_unique_customers,
    base.cancellation_rate,
    base.avg_amount,
    base.amount_std,
    concentration.top3_customer_concentration
FROM base
JOIN concentration ON base.seller_id = concentration.seller_id
WHERE base.n_events >= 3;
"""

cursor.execute(seller_features_query)
rows = cursor.fetchall()
columns = [d[0] for d in cursor.description]
feats = pd.DataFrame(rows, columns=columns)

print(f"Продавцов с 3+ заказами: {len(feats)}")
print()

rule_flag_query = """
WITH order_seller AS (
    SELECT oi.seller_id, o.customer_id, o.order_status
    FROM olist_order_items_dataset oi
    JOIN olist_orders_dataset o ON oi.order_id = o.order_id
)
SELECT
    seller_id,
    COUNT(*) AS n_events,
    COUNT(DISTINCT customer_id) AS n_unique_customers,
    AVG(CASE WHEN order_status IN ('canceled', 'unavailable') THEN 1.0 ELSE 0.0 END) AS cancellation_rate
FROM order_seller
GROUP BY seller_id
HAVING n_unique_customers <= 2 AND cancellation_rate >= 0.9 AND n_events >= 3;
"""

cursor.execute(rule_flag_query)
rule_rows = cursor.fetchall()
print(f"Продавцов, у которых почти все заказы от 1-2 покупателей и почти всё отменено: {len(rule_rows)}")
for seller_id, n_events, n_customers, cancel_rate in rule_rows:
    print(f"  {seller_id} - {n_events} заказов, {n_customers} покупателей, отмен {cancel_rate:.0%}")

conn.close()

feats["log_n_events"] = feats["n_events"].apply(lambda x: math.log(x + 1))
feats["log_avg_amount"] = feats["avg_amount"].clip(lower=0).apply(lambda x: math.log(x + 1))
feats["amount_std"] = feats["amount_std"].fillna(0)

model_columns = [
    "log_n_events", "n_unique_customers", "top3_customer_concentration",
    "cancellation_rate", "log_avg_amount", "amount_std",
]
X = feats[model_columns].fillna(0)

model = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
feats["anomaly_score"] = model.fit_predict(X)
feats["anomaly_score_raw"] = model.decision_function(X)

anomalies = feats[feats["anomaly_score"] == -1].sort_values("anomaly_score_raw")
print()
print(f"Isolation Forest пометил {len(anomalies)} продавцов как аномальных из {len(feats)}")

show_cols = ["seller_id", "n_events", "n_unique_customers", "top3_customer_concentration", "cancellation_rate", "avg_amount"]
print(anomalies[show_cols].head(15).to_string(index=False))

anomalies.to_csv(report_dir / "olist_anomalous_sellers.csv", index=False)
feats.to_csv(report_dir / "olist_all_seller_features.csv", index=False)
print()
print(f"Сохранил результаты в {report_dir}")
