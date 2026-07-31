import pandas as pd
import numpy as np


def build_entity_features(unified, reference_date=None):
    df = unified.copy()
    has_timestamp = df["timestamp"].notna().any()

    if has_timestamp:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if reference_date is None:
            reference_date = df["timestamp"].max()

    grouped = df.groupby("entity_id")
    feats = pd.DataFrame(index=grouped.size().index)

    feats["n_events"] = grouped.size()
    feats["n_unique_customers"] = grouped["customer_id"].nunique()

    if has_timestamp:
        feats["first_event_ts"] = grouped["timestamp"].min()
        feats["last_event_ts"] = grouped["timestamp"].max()
        feats["lifetime_days"] = (feats["last_event_ts"] - feats["first_event_ts"]).dt.total_seconds() / 86400
        feats["lifetime_days"] = feats["lifetime_days"].clip(lower=1)
        feats["events_per_day"] = feats["n_events"] / feats["lifetime_days"]

        feats["days_since_first_event"] = (reference_date - feats["first_event_ts"]).dt.total_seconds() / 86400
        early_window = df.merge(feats[["first_event_ts"]], left_on="entity_id", right_index=True)
        early_window["is_early"] = (
            early_window["timestamp"] - early_window["first_event_ts"]
        ).dt.total_seconds() / 86400 <= 14
        feats["pct_events_in_first_14d"] = early_window.groupby("entity_id")["is_early"].mean()

    def top3_share(g):
        counts = g.value_counts()
        return counts.head(3).sum() / counts.sum() if counts.sum() > 0 else np.nan

    feats["top3_customer_concentration"] = grouped["customer_id"].apply(top3_share)
    feats["cancellation_rate"] = grouped["is_cancelled"].mean()

    if df["amount"].notna().any():
        feats["total_amount"] = grouped["amount"].sum()
        feats["avg_amount"] = grouped["amount"].mean()
        feats["amount_std"] = grouped["amount"].std().fillna(0)

    if df["category"].notna().any():
        cat_baseline = df.groupby("category")["is_cancelled"].mean().rename("category_baseline_cancel_rate")
        entity_primary_cat = grouped["category"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
        feats = feats.join(entity_primary_cat.rename("primary_category"))
        feats = feats.merge(cat_baseline, left_on="primary_category", right_index=True, how="left")
        feats["cancel_rate_deviation"] = feats["cancellation_rate"] - feats["category_baseline_cancel_rate"]

    feats["source"] = df["source"].iloc[0] if len(df) else pd.NA

    feats["log_n_events"] = np.log1p(feats["n_events"])
    if "avg_amount" in feats.columns:
        feats["log_avg_amount"] = np.log1p(feats["avg_amount"].clip(lower=0))
    if "amount_std" in feats.columns:
        feats["log_amount_std"] = np.log1p(feats["amount_std"].clip(lower=0))

    feats["rule_flag_single_customer_all_cancelled"] = (
        (feats["n_unique_customers"] <= 2) &
        (feats["cancellation_rate"] >= 0.9) &
        (feats["n_events"] >= 3)
    )

    return feats.reset_index()


NUMERIC_FEATURES_FOR_MODEL = [
    "log_n_events", "n_unique_customers", "events_per_day",
    "pct_events_in_first_14d", "top3_customer_concentration",
    "cancellation_rate", "log_avg_amount", "log_amount_std", "cancel_rate_deviation",
]


def select_model_features(feats):
    available = [c for c in NUMERIC_FEATURES_FOR_MODEL if c in feats.columns]
    X = feats[available].copy()
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    return X


def split_by_size(feats, small_threshold=None):
    if small_threshold is None:
        small_threshold = feats["n_events"].median()
    small = feats[feats["n_events"] <= small_threshold].reset_index(drop=True)
    large = feats[feats["n_events"] > small_threshold].reset_index(drop=True)
    return small, large
