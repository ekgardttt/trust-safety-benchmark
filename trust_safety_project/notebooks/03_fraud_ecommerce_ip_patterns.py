import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data" / "fraud_ecommerce"
report_dir = project_root / "report"
report_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_dir / "Fraud_Data.csv", parse_dates=["signup_time", "purchase_time"])
print(f"Загружено транзакций: {len(df)}")
print(f"Реальный fraud rate в датасете: {df['class'].mean():.2%}\n")

df["signup_to_purchase_seconds"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()

device_agg = df.groupby("device_id").agg(
    n_users=("user_id", "nunique"),
    n_transactions=("user_id", "count"),
    fraud_rate=("class", "mean"),
    avg_signup_to_purchase=("signup_to_purchase_seconds", "mean"),
).reset_index()

device_agg = device_agg.sort_values("n_users", ascending=False)

print("=== Топ-15 устройств по числу разных аккаунтов (device sharing) ===")
print(device_agg.head(15).to_string(index=False))

multi_account_devices = device_agg[device_agg["n_users"] >= 2]
print(f"\nУстройств с >=2 аккаунтами: {len(multi_account_devices)}")
print(f"Средний fraud_rate на таких устройствах: {multi_account_devices['fraud_rate'].mean():.2%}")
print(f"Средний fraud_rate на устройствах с 1 аккаунтом: {device_agg[device_agg['n_users']==1]['fraud_rate'].mean():.2%}")

ip_country = pd.read_csv(data_dir / "IpAddress_to_Country.csv")
ip_country = ip_country.sort_values("lower_bound_ip_address").reset_index(drop=True)

df_sorted = df.sort_values("ip_address").reset_index(drop=True)
merged = pd.merge_asof(
    df_sorted,
    ip_country,
    left_on="ip_address",
    right_on="lower_bound_ip_address",
    direction="backward",
)
merged = merged[merged["ip_address"] <= merged["upper_bound_ip_address"]]

ip_agg = df.groupby("ip_address").agg(
    n_users=("user_id", "nunique"),
    n_transactions=("user_id", "count"),
    fraud_rate=("class", "mean"),
).reset_index().sort_values("n_users", ascending=False)

print(f"\n=== Топ-15 IP-адресов по числу разных аккаунтов ===")
print(ip_agg.head(15).to_string(index=False))

multi_account_ips = ip_agg[ip_agg["n_users"] >= 2]
print(f"\nIP с >=2 аккаунтами: {len(multi_account_ips)}")
print(f"Средний fraud_rate на таких IP: {multi_account_ips['fraud_rate'].mean():.2%}")
print(f"Средний fraud_rate на IP с 1 аккаунтом: {ip_agg[ip_agg['n_users']==1]['fraud_rate'].mean():.2%}")

quick_signup = df[df["signup_to_purchase_seconds"] < 60]
print(f"\n=== Покупки в первую минуту после регистрации ===")
print(f"Таких транзакций: {len(quick_signup)} из {len(df)} ({len(quick_signup)/len(df):.2%})")
print(f"Fraud rate среди них: {quick_signup['class'].mean():.2%}")
print(f"Fraud rate в среднем по датасету: {df['class'].mean():.2%}")

device_agg.to_csv(report_dir / "fraud_ecommerce_device_patterns.csv", index=False)
ip_agg.to_csv(report_dir / "fraud_ecommerce_ip_patterns.csv", index=False)
merged.to_csv(report_dir / "fraud_ecommerce_with_country.csv", index=False)
print(f"\nСохранено в {report_dir}")
