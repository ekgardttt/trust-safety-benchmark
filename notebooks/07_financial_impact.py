import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data" / "fraud_ecommerce"
report_dir = project_root / "report"
report_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_dir / "Fraud_Data.csv", parse_dates=["signup_time", "purchase_time"])
df = df.sort_values("purchase_time").reset_index(drop=True)

total_fraud_revenue = df.loc[df["class"] == 1, "purchase_value"].sum()
total_revenue = df["purchase_value"].sum()
print(f"Всего транзакций: {len(df)}")
print(f"Общая выручка: {total_revenue:,.2f}")
print(f"Выручка от мошеннических транзакций: {total_fraud_revenue:,.2f} ({total_fraud_revenue/total_revenue:.2%} от общей)")
print()

def simulate_detection(df, threshold):
    df = df.copy()
    df["seen_users_on_device"] = df.groupby("device_id")["user_id"].transform(
        lambda s: s.expanding().apply(lambda x: pd.Series(x).nunique())
    )
    df["detected"] = df["seen_users_on_device"] >= threshold
    prevented_mask = df["detected"] & (df["class"] == 1)
    preventable_revenue = df.loc[prevented_mask, "purchase_value"].sum()
    preventable_count = prevented_mask.sum()
    return preventable_revenue, preventable_count

results = []
for threshold in [2, 3, 5, 10]:
    revenue, count = simulate_detection(df, threshold)
    results.append({
        "detection_threshold_accounts": threshold,
        "preventable_fraud_transactions": count,
        "preventable_fraud_revenue": revenue,
        "pct_of_total_fraud_revenue": revenue / total_fraud_revenue,
    })
    print(f"Порог блокировки: {threshold} аккаунтов на устройство")
    print(f"  Предотвратимых мошеннических транзакций: {count}")
    print(f"  Предотвратимая выручка: {revenue:,.2f} ({revenue/total_fraud_revenue:.2%} от всей мошеннической выручки)")
    print()

results_df = pd.DataFrame(results)
results_df.to_csv(report_dir / "financial_impact_by_threshold.csv", index=False)

print(" Вывод ")
fastest = results_df.iloc[0]["preventable_fraud_revenue"]
slowest = results_df.iloc[-1]["preventable_fraud_revenue"]
print(f"Разница между порогом 2 и порогом 10 аккаунтов: {fastest - slowest:,.2f} в выручке")
print(f"Сохранено: {report_dir / 'financial_impact_by_threshold.csv'}")
