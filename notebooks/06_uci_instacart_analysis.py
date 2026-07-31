import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.ensemble import IsolationForest

from schema_mapping import load_uci_retail, load_instacart
from shared_features import build_entity_features, select_model_features

project_root = Path(__file__).resolve().parent.parent
report_dir = project_root / "report"
report_dir.mkdir(exist_ok=True)


def run_company(name, unified, min_events=3):
    print(f"\n {name} ")
    print(f"Событий: {len(unified)}, уникальных клиентов: {unified['entity_id'].nunique()}")

    feats = build_entity_features(unified)
    feats = feats[feats["n_events"] >= min_events].reset_index(drop=True)
    print(f"Клиентов с >={min_events} событиями: {len(feats)}")

    X = select_model_features(feats)
    model = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
    feats["anomaly_score"] = model.fit_predict(X)
    feats["anomaly_score_raw"] = model.decision_function(X)

    anomalies = feats[feats["anomaly_score"] == -1].sort_values("anomaly_score_raw")
    print(f"Аномалий: {len(anomalies)} ({len(anomalies)/len(feats):.1%})")

    cols = [
        "entity_id", "n_events", "n_unique_customers", "top3_customer_concentration",
        "cancellation_rate", "avg_amount", "anomaly_score_raw",
    ]
    cols = [c for c in cols if c in anomalies.columns]
    print(anomalies[cols].head(10).to_string(index=False))

    safe_name = name.lower().replace(" ", "_")
    anomalies.to_csv(report_dir / f"{safe_name}_anomalies.csv", index=False)
    feats.to_csv(report_dir / f"{safe_name}_all_features.csv", index=False)
    return anomalies


uci_unified = load_uci_retail(str(project_root / "data" / "uci_retail"))
run_company("uci_retail", uci_unified, min_events=3)

instacart_unified = load_instacart(str(project_root / "data" / "instacart"))
run_company("instacart", instacart_unified, min_events=5)

print("\nГотово. Результаты сохранены в report/")
