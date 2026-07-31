import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
report_dir = project_root / "report"

df = pd.read_csv(report_dir / "fraud_ecommerce_with_country.csv")
df["country"] = df["country"].fillna("Unknown")

country_agg = df.groupby("country").agg(
    n_transactions=("class", "count"),
    fraud_rate=("class", "mean"),
    avg_purchase_value=("purchase_value", "mean"),
).reset_index()

country_agg = country_agg.sort_values("n_transactions", ascending=False)

print(" Топ-20 стран по объёму транзакций ")
print(country_agg.head(20).to_string(index=False))

reliable = country_agg[country_agg["n_transactions"] >= 100]
reliable_sorted = reliable.sort_values("fraud_rate", ascending=False)

print("\n Топ-20 стран по  доле мошеннических транзакций (>=100 транзакций) ")
print(reliable_sorted.head(20).to_string(index=False))

print("\n Страны с наименьшей долей мошеннических транзакций (>=100 транзакций) ")
print(reliable_sorted.tail(10).to_string(index=False))

overall_rate = df["class"].mean()
print(f"\nСредняя доля мошеннических транзакций по всему датасету: {overall_rate:.2%}")

country_agg.to_csv(report_dir / "fraud_ecommerce_by_country.csv", index=False)
print(f"\nСохранено: {report_dir / 'fraud_ecommerce_by_country.csv'}")
