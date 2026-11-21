"""
Customer segmentation pipeline (feature engineering + K-means).

Shared by the data generator (to decide who gets the coupon program on
2023-09-01, exactly as the business did) and by notebooks/03_customer_segmentation.ipynb
(to reproduce and explain the segments).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_START = pd.Timestamp("2023-02-01")
PROMO_WINDOWS = [("2023-06-01", "2023-06-18"), ("2023-10-24", "2023-11-11")]

FEATURES = [
    "recency_days",       # days since last visit
    "visits_per_30d",     # visit frequency
    "orders_per_30d",     # purchase frequency
    "avg_order_value",    # monetary
    "visit_to_order",     # orders / visits
    "pv_per_visit",       # browsing depth
    "coupon_share",       # share of orders using a coupon
    "promo_order_share",  # share of orders placed in mega-sale windows
    "tenure_days",        # days since signup (capped at data start)
]
LOG_FEATURES = ["recency_days", "visits_per_30d", "orders_per_30d", "avg_order_value", "tenure_days"]


def in_promo(dates: pd.Series) -> pd.Series:
    mask = pd.Series(False, index=dates.index)
    for a, b in PROMO_WINDOWS:
        mask |= dates.between(pd.Timestamp(a), pd.Timestamp(b))
    return mask


def build_features(behavior, orders, customers, cutoff):
    """One row per customer with at least one visit up to `cutoff`."""
    cutoff = pd.Timestamp(cutoff)
    b = behavior[pd.to_datetime(behavior["date"]) <= cutoff].copy()
    b["date"] = pd.to_datetime(b["date"])
    o = orders[(pd.to_datetime(orders["order_date"]) <= cutoff)].copy()
    if "order_status" in o:
        o = o[o["order_status"] == "completed"]
    o["order_date"] = pd.to_datetime(o["order_date"])
    o["promo"] = in_promo(o["order_date"])

    v = b.groupby("customer_id").agg(
        visits=("date", "size"), page_views=("page_views", "sum"), last_visit=("date", "max"))
    f = o.groupby("customer_id").agg(
        orders=("order_date", "size"), avg_order_value=("net_amount", "mean"),
        coupon_share=("coupon_used", "mean"), promo_order_share=("promo", "mean"))

    c = customers[["customer_id", "signup_date"]].copy()
    c["signup_date"] = pd.to_datetime(c["signup_date"])
    x = c.merge(v, on="customer_id", how="inner").merge(f, on="customer_id", how="left")
    x["orders"] = x["orders"].fillna(0)
    for col in ["avg_order_value", "coupon_share", "promo_order_share"]:
        x[col] = x[col].fillna(0)
    x["tenure_days"] = (cutoff - x["signup_date"].clip(lower=DATA_START)).dt.days.clip(lower=14)
    x["recency_days"] = (cutoff - x["last_visit"]).dt.days
    x["visits_per_30d"] = x["visits"] / x["tenure_days"] * 30
    x["orders_per_30d"] = x["orders"] / x["tenure_days"] * 30
    x["visit_to_order"] = x["orders"] / x["visits"]
    x["pv_per_visit"] = x["page_views"] / x["visits"]
    return x.set_index("customer_id")


def transform(features: pd.DataFrame) -> np.ndarray:
    X = features[FEATURES].copy()
    for col in LOG_FEATURES:
        X[col] = np.log1p(X[col])
    return StandardScaler().fit_transform(X)


def fit_segments(features: pd.DataFrame, k: int = 5, seed: int = 0):
    Xs = transform(features)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Xs)
    labels = pd.Series(km.labels_, index=features.index, name="cluster")
    return labels, km, Xs


def name_clusters(features: pd.DataFrame, labels: pd.Series) -> dict:
    """Give each cluster a business name from its median profile (rule-based, transparent)."""
    prof = features.assign(cluster=labels).groupby("cluster")[FEATURES].median()
    names, left = {}, list(prof.index)

    def take(col, name, largest=True):
        cand = prof.loc[left, col]
        cl = cand.idxmax() if largest else cand.idxmin()
        names[cl] = name
        left.remove(cl)

    take("orders_per_30d", "High-Value Loyal")
    take("recency_days", "Dormant / At-Risk")
    take("tenure_days", "New Potential", largest=False)
    take("coupon_share", "Deal Seekers")
    for cl in left:
        names[cl] = "Window Shoppers"
    return names


TARGET_SEGMENTS = ["Deal Seekers", "New Potential"]
