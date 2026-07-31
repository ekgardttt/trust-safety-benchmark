import sys
import sqlite3
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
db_path = project_root / "olist_db.db"
report_dir = project_root / "report"

flagged_files = [
    "olist_rule_flagged_sellers.csv",
    "olist_small_seller_anomalies.csv",
    "olist_large_seller_anomalies.csv",
]

entity_ids = set()
for fname in flagged_files:
    fpath = report_dir / fname
    if fpath.exists():
        df = pd.read_csv(fpath)
        entity_ids.update(df["entity_id"].astype(str).tolist())

entity_ids = list(entity_ids)
print(f"Проверяем отзывы для {len(entity_ids)} flagged продавцов\n")

conn = sqlite3.connect(str(db_path))
placeholders = ",".join(["?"] * len(entity_ids))

query = f"""
SELECT
    oi.seller_id,
    o.order_id,
    o.order_status,
    r.review_score
FROM olist_order_items_dataset oi
JOIN olist_orders_dataset o ON oi.order_id = o.order_id
LEFT JOIN olist_order_reviews_dataset r ON o.order_id = r.order_id
WHERE oi.seller_id IN ({placeholders})
"""

df = pd.read_sql_query(query, conn, params=entity_ids)
conn.close()

df["is_cancelled"] = df["order_status"].isin(["canceled", "unavailable"])
df["is_5star"] = df["review_score"] == 5

summary = df.groupby("seller_id").agg(
    n_orders=("order_id", "nunique"),
    n_reviews=("review_score", "count"),
    avg_review_score=("review_score", "mean"),
    pct_5star=("is_5star", "mean"),
    cancellation_rate=("is_cancelled", "mean"),
).reset_index()

cancelled_with_review = df[df["is_cancelled"] & df["review_score"].notna()]
cancelled_5star = cancelled_with_review[cancelled_with_review["review_score"] == 5]
cancelled_summary = cancelled_5star.groupby("seller_id").size().rename("n_5star_on_cancelled_orders")
summary = summary.merge(cancelled_summary, left_on="seller_id", right_index=True, how="left")
summary["n_5star_on_cancelled_orders"] = summary["n_5star_on_cancelled_orders"].fillna(0).astype(int)

summary = summary.sort_values("n_5star_on_cancelled_orders", ascending=False)

print(summary.to_string(index=False))

out_path = report_dir / "olist_flagged_sellers_review_check.csv"
summary.to_csv(out_path, index=False)
print(f"\nСохранено: {out_path}")
