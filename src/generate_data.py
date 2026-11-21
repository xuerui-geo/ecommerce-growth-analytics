"""
Synthetic data generator for the E-commerce Growth Analytics project.

Simulates a mid-sized online store on a Taobao/Tmall-style marketplace between
2023-02-01 and 2023-11-30. Field structure is modelled on Alibaba's public
"Ad Display/Click Data on Taobao.com" dataset (user profile + ad campaign
tables) and on typical marketplace "Business Advisor" store reports
(UV / PV / add-to-cart / orders by traffic source).

All data is synthetic. No real company, customer or campaign data is used.

Planted business scenarios (for the analysis to discover):
  1. Seasonality: 618 mid-year sale (Jun 1-18) and Singles' Day (Oct 24 - Nov 11).
  2. Campaign anomaly: campaign C03 (display retargeting) had its audience
     targeting mis-configured from 2023-07-12 to 2023-07-30 -- impressions
     drifted to low-fit audiences, so CTR / CVR / ROAS fell while
     within-audience CTR stayed stable (a mix-shift problem).
  3. Tracking outage: C02 click tracking failed on 2023-04-15 (clicks = 0
     while impressions and spend are normal) -- a data-quality issue.
  4. Hidden customer segments with different behaviour, to be recovered by
     clustering.
  5. From 2023-09-01 a segment-targeted coupon program was rolled out to
     customers in two segments, with a random 20% holdout (control group),
     so its impact can be measured as an experiment.

Usage:  python src/generate_data.py   (writes CSVs to data/raw/)
"""

from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segmentation as seg_lib  # noqa: E402

SEED = 2023
rng = np.random.default_rng(SEED)

START = pd.Timestamp("2023-02-01")
END = pd.Timestamp("2023-11-30")
DATES = pd.date_range(START, END, freq="D")
N_DAYS = len(DATES)
N_CUSTOMERS = 15_000
PROGRAM_START = pd.Timestamp("2023-09-01")

OUT = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

AGE_GROUPS = ["18-24", "25-29", "30-34", "35-44", "45+"]
CITY_TIERS = ["Tier 1", "Tier 2", "Tier 3", "Tier 4+"]
CATEGORIES = ["Snacks", "Tea & Coffee", "Health Food", "Gift Boxes", "Pantry"]


# ---------------------------------------------------------------- calendar
def day_multipliers(dates):
    """Traffic multiplier and promo flag for each date."""
    traffic = np.ones(len(dates))
    promo = np.zeros(len(dates), dtype=bool)
    for i, d in enumerate(dates):
        if d.dayofweek >= 5:
            traffic[i] *= 1.15
        if pd.Timestamp("2023-06-01") <= d <= pd.Timestamp("2023-06-18"):
            traffic[i] *= 1.3
            promo[i] = True
        if d == pd.Timestamp("2023-06-18"):
            traffic[i] *= 2.0
        if pd.Timestamp("2023-10-24") <= d <= pd.Timestamp("2023-11-11"):
            traffic[i] *= 1.45
            promo[i] = True
        if d == pd.Timestamp("2023-11-11"):
            traffic[i] *= 2.8
    return traffic, promo


TRAFFIC_MULT, PROMO = day_multipliers(DATES)


# ---------------------------------------------------------------- customers
SEGMENTS = {
    # visit: base daily visit prob | buy: order prob per visit | aov: median order value (CNY)
    # promo_lift / promo_visit: response to mega-sales | coupon: extra coupon propensity
    "loyal_high_value": dict(share=0.12, visit=0.10, buy=0.22, aov=175, promo_lift=1.2, promo_visit=1.0,
                             coupon=0.00, pv=2.5, fav=0.10, cart=0.18,
                             age=[.05, .25, .35, .25, .10], tier=[.40, .35, .17, .08]),
    "deal_seeker":      dict(share=0.26, visit=0.06, buy=0.07, aov=60, promo_lift=3.0, promo_visit=2.2,
                             coupon=0.40, pv=2.5, fav=0.08, cart=0.20,
                             age=[.35, .30, .15, .12, .08], tier=[.10, .25, .35, .30]),
    "new_potential":    dict(share=0.20, visit=0.09, buy=0.08, aov=100, promo_lift=1.6, promo_visit=1.3,
                             coupon=0.10, pv=2.8, fav=0.12, cart=0.16,
                             age=[.25, .35, .22, .12, .06], tier=[.25, .35, .25, .15]),
    "window_shopper":   dict(share=0.24, visit=0.13, buy=0.012, aov=80, promo_lift=1.5, promo_visit=1.2,
                             coupon=0.05, pv=4.5, fav=0.20, cart=0.10,
                             age=[.30, .28, .20, .14, .08], tier=[.22, .30, .28, .20]),
    "dormant":          dict(share=0.18, visit=0.07, buy=0.07, aov=115, promo_lift=1.3, promo_visit=1.0,
                             coupon=0.05, pv=2.5, fav=0.08, cart=0.14,
                             age=[.08, .15, .25, .30, .22], tier=[.25, .30, .25, .20]),
}
SEG_NAMES = list(SEGMENTS)

CATEGORY_PREF = {
    "loyal_high_value": [.15, .30, .20, .25, .10],
    "deal_seeker":      [.45, .10, .10, .05, .30],
    "new_potential":    [.30, .20, .25, .10, .15],
    "window_shopper":   [.35, .20, .15, .15, .15],
    "dormant":          [.20, .25, .20, .15, .20],
}

SOURCE_NAMES = ["Organic Search", "Paid Search", "Display Ads", "Recommendation Feed",
                "Favorites/Direct", "Livestream/Influencer"]
SOURCE_PREF = {
    "loyal_high_value": [.15, .10, .05, .15, .50, .05],
    "deal_seeker":      [.20, .15, .15, .25, .05, .20],
    "new_potential":    [.25, .20, .20, .25, .05, .05],
    "window_shopper":   [.20, .10, .15, .45, .05, .05],
    "dormant":          [.25, .10, .10, .20, .30, .05],
}

seg = rng.choice(SEG_NAMES, size=N_CUSTOMERS, p=[SEGMENTS[s]["share"] for s in SEG_NAMES])
age = np.array([rng.choice(AGE_GROUPS, p=SEGMENTS[s]["age"]) for s in seg])
tier = np.array([rng.choice(CITY_TIERS, p=SEGMENTS[s]["tier"]) for s in seg])
gender = rng.choice(["F", "M"], size=N_CUSTOMERS, p=[0.64, 0.36]).astype(object)
gender[rng.random(N_CUSTOMERS) < 0.05] = None  # ~5% missing, like the public Taobao profile data

# signup dates: new_potential sign up during the window, others mostly before it
signup = np.empty(N_CUSTOMERS, dtype="datetime64[ns]")
pre_start, pre_days = pd.Timestamp("2021-06-01"), (START - pd.Timestamp("2021-06-01")).days
for i, s in enumerate(seg):
    if s == "new_potential":
        signup[i] = START + pd.Timedelta(days=int(rng.integers(0, N_DAYS - 30)))
    elif rng.random() < 0.15 and s != "dormant":
        signup[i] = START + pd.Timedelta(days=int(rng.integers(0, N_DAYS - 30)))
    else:
        signup[i] = pre_start + pd.Timedelta(days=int(rng.integers(0, pre_days)))

acq_channel = np.array([rng.choice(SOURCE_NAMES, p=SOURCE_PREF[s]) for s in seg])
device = rng.choice(["Mobile App", "Mobile Web", "PC"], size=N_CUSTOMERS, p=[.78, .12, .10])
customer_ids = np.array([f"U{100000 + i}" for i in range(N_CUSTOMERS)])

customers = pd.DataFrame({
    "customer_id": customer_ids,
    "signup_date": pd.to_datetime(signup).date,
    "gender": gender,
    "age_group": age,
    "city_tier": tier,
    "device_preference": device,
    "acquisition_channel": acq_channel,
})

# ---------------------------------------------------------------- daily behaviour
# Phase 1: simulate all visits (visits are not affected by the coupon program).
signup_idx = ((pd.to_datetime(signup) - START).days).to_numpy()
day_ix = np.arange(N_DAYS)
visit_parts = []
for s in SEG_NAMES:
    p = SEGMENTS[s]
    idx = np.where(seg == s)[0]
    n = len(idx)
    visit_rate = p["visit"] * rng.lognormal(0, 0.4, n)
    buy_rate = np.clip(p["buy"] * rng.lognormal(0, 0.3, n), 0, 0.6)
    rate = np.outer(visit_rate, TRAFFIC_MULT)
    rate *= np.where(PROMO, p["promo_visit"], 1.0)[None, :]
    rate[day_ix[None, :] < signup_idx[idx][:, None]] = 0
    if s == "dormant":
        churn_day = rng.integers(20, 110, n)
        rate *= np.where(day_ix[None, :] > churn_day[:, None], 0.05, 1.0)
    visits = rng.random(rate.shape) < np.clip(rate, 0, 0.95)
    ci, di = np.nonzero(visits)
    m = len(ci)
    visit_parts.append(pd.DataFrame({
        "gi": idx[ci], "di": di, "seg": s, "buy_rate": buy_rate[ci],
        "pv": 1 + rng.poisson(p["pv"], m),
        "fav": rng.binomial(1, p["fav"], m),
        "cart": rng.binomial(1, p["cart"], m),
        "src": rng.choice(len(SOURCE_NAMES), size=m, p=SOURCE_PREF[s]),
    }))
V = pd.concat(visit_parts, ignore_index=True)
V["date"] = DATES[V["di"].to_numpy()]
V["customer_id"] = customer_ids[V["gi"].to_numpy()]
V["promo"] = PROMO[V["di"].to_numpy()]


def simulate_orders(Vp, treated_mask):
    """Number of orders per visit and order lines for the visits in Vp."""
    segs = Vp["seg"].to_numpy()
    promo_lift = np.array([SEGMENTS[s]["promo_lift"] for s in segs])
    pbuy = Vp["buy_rate"].to_numpy().copy()
    pbuy *= np.where(Vp["promo"].to_numpy(), promo_lift, 1.0)
    pbuy *= np.where(Vp["cart"].to_numpy() == 1, 2.2, 0.8)
    # coupon program effect: strong for price-sensitive / new customers, weak for others
    responsive = np.isin(segs, ["deal_seeker", "new_potential"])
    pbuy *= np.where(treated_mask, np.where(responsive, 1.55, 1.15), 1.0)
    pbuy = np.clip(pbuy, 0, 0.9)
    n_orders = rng.binomial(1, pbuy) + rng.binomial(1, pbuy * 0.08)

    oi = np.repeat(np.arange(len(Vp)), n_orders)
    k = len(oi)
    oseg = segs[oi]
    cat = np.array([rng.choice(CATEGORIES, p=CATEGORY_PREF[s]) for s in oseg])
    items = 1 + rng.poisson(np.where(oseg == "deal_seeker", 1.1, 0.6))
    aov = np.array([SEGMENTS[s]["aov"] for s in oseg])
    gross = np.round(aov * rng.lognormal(0, 0.4, k) * (1 + 0.15 * (items - 1)), 2)
    promo_day = Vp["promo"].to_numpy()[oi]
    coupon_prob = (np.where(promo_day, 0.45, 0.10)
                   + np.array([SEGMENTS[s]["coupon"] for s in oseg])
                   + np.where(treated_mask[oi], 0.35, 0))
    coupon = rng.random(k) < np.clip(coupon_prob, 0, 0.95)
    discount = np.where(coupon, np.round(gross * rng.uniform(0.05, 0.2, k), 2), 0.0)
    discount = np.where(promo_day, discount + np.round(gross * 0.05, 2), discount)
    lines = pd.DataFrame({
        "customer_id": Vp["customer_id"].to_numpy()[oi],
        "order_date": Vp["date"].dt.date.to_numpy()[oi],
        "traffic_source": np.array(SOURCE_NAMES)[Vp["src"].to_numpy()[oi]],
        "category": cat, "items": items, "gross_amount": gross,
        "discount_amount": discount, "coupon_used": coupon.astype(int),
    })
    return n_orders, lines


# Phase 2: orders before the program launch
pre = V["date"] < PROGRAM_START
n_pre, orders_pre = simulate_orders(V[pre], np.zeros(pre.sum(), dtype=bool))
orders_pre["order_status"] = np.where(rng.random(len(orders_pre)) < 0.015, "refunded", "completed")
orders_pre["net_amount"] = (orders_pre["gross_amount"] - orders_pre["discount_amount"]).round(2)

# Phase 3: on 2023-08-31 the business clusters customers on Feb-Aug data and picks
# the target segments -- run the same pipeline the notebook uses.
beh_pre = V[pre].rename(columns={"pv": "page_views"})[["customer_id", "date", "page_views"]]
cust_tmp = pd.DataFrame({"customer_id": customer_ids, "signup_date": pd.to_datetime(signup)})
feats = seg_lib.build_features(beh_pre, orders_pre, cust_tmp, "2023-08-31")
labels, _, _ = seg_lib.fit_segments(feats)
names = seg_lib.name_clusters(feats, labels)
seg_name = labels.map(names)
eligible_ids = seg_name[seg_name.isin(seg_lib.TARGET_SEGMENTS)].index.to_numpy()
treat_flag = rng.random(len(eligible_ids)) < 0.8
coupon_program = pd.DataFrame({
    "customer_id": eligible_ids,
    "segment_at_launch": seg_name.loc[eligible_ids].to_numpy(),
    "program": "Segment-Targeted Coupon Program",
    "launch_date": PROGRAM_START.date(),
    "group": np.where(treat_flag, "treatment", "control"),
})
treated_ids = set(eligible_ids[treat_flag])

# Phase 4: orders after launch
Vpost = V[~pre]
treated_mask = Vpost["customer_id"].isin(treated_ids).to_numpy()
n_post, orders_post = simulate_orders(Vpost, treated_mask)
orders_post["order_status"] = np.where(rng.random(len(orders_post)) < 0.015, "refunded", "completed")
orders_post["net_amount"] = (orders_post["gross_amount"] - orders_post["discount_amount"]).round(2)

V.loc[pre, "orders"] = n_pre
V.loc[~pre, "orders"] = n_post
behaviour = pd.DataFrame({
    "customer_id": V["customer_id"], "date": V["date"].dt.date,
    "traffic_source": np.array(SOURCE_NAMES)[V["src"].to_numpy()],
    "page_views": V["pv"], "favorites": V["fav"], "add_to_cart": V["cart"],
    "orders": V["orders"].astype(int),
}).sort_values(["date", "customer_id"]).reset_index(drop=True)

orders = (pd.concat([orders_pre, orders_post]).sort_values(["order_date", "customer_id"])
          .reset_index(drop=True))
orders.insert(0, "order_id", [f"O{2300000 + i}" for i in range(len(orders))])
orders = orders[["order_id", "customer_id", "order_date", "traffic_source", "category", "items",
                 "gross_amount", "discount_amount", "coupon_used", "net_amount", "order_status"]]

# ---------------------------------------------------------------- ad campaigns
AUD = [(a, t) for a in AGE_GROUPS for t in CITY_TIERS]
# audience "fit" for the store's products: CTR / CVR multiplier
AGE_FIT = dict(zip(AGE_GROUPS, [0.75, 1.15, 1.25, 1.0, 0.7]))
TIER_FIT = dict(zip(CITY_TIERS, [1.2, 1.1, 0.85, 0.65]))


def weights(age_w, tier_w):
    w = np.array([age_w[AGE_GROUPS.index(a)] * tier_w[CITY_TIERS.index(t)] for a, t in AUD])
    return w / w.sum()


CAMPAIGNS = [
    # id, name, type, base CTR, base CVR, CPC, daily budget, start, end, targeting
    ("C01", "Brand Keywords", "Paid Search", .060, .085, 0.9, 900, START, END,
     weights([1, 1, 1, 1, 1], [1, 1, 1, 1])),
    ("C02", "Category Keywords", "Paid Search", .034, .036, 1.4, 1800, START, END,
     weights([1, 1.2, 1.2, 1, .8], [1.2, 1.1, 1, .8])),
    ("C03", "Visitor Retargeting", "Display Ads", .029, .048, 1.1, 1200, START, END,
     weights([.4, 1.6, 1.8, 1, .4], [1.6, 1.3, .6, .3])),
    ("C04", "Interest Audience Prospecting", "Display Ads", .015, .016, 0.7, 1500, START, END,
     weights([1.3, 1.2, 1, .9, .6], [1, 1, 1, 1])),
    ("C05", "New Product Launch - Health Food", "Smart Campaign", .022, .025, 1.0, 1000,
     pd.Timestamp("2023-04-01"), pd.Timestamp("2023-06-30"),
     weights([.6, 1.2, 1.4, 1.2, .8], [1.4, 1.2, .8, .5])),
    ("C06", "Mega-Sale Livestream Boost", "Livestream/Influencer", .041, .052, 1.6, 3000,
     pd.Timestamp("2023-06-01"), pd.Timestamp("2023-06-18"),
     weights([1.4, 1.3, 1, .7, .4], [1, 1, 1, 1])),
    ("C07", "Singles' Day Livestream Boost", "Livestream/Influencer", .043, .055, 1.6, 3500,
     pd.Timestamp("2023-10-24"), pd.Timestamp("2023-11-11"),
     weights([1.4, 1.3, 1, .7, .4], [1, 1, 1, 1])),
    ("C08", "Segment-Targeted Lookalike", "Smart Campaign", .031, .044, 1.2, 1500,
     PROGRAM_START, END,
     weights([1.1, 1.5, 1.3, .8, .4], [1.3, 1.3, .9, .6])),
]

ANOMALY_START, ANOMALY_END = pd.Timestamp("2023-07-12"), pd.Timestamp("2023-07-30")
C03_BROKEN = weights([3.0, .8, .6, .5, .6], [.4, .6, 1.4, 2.4])  # drifted to 18-24 & lower tiers
OUTAGE_DAY = pd.Timestamp("2023-04-15")

ad_rows = []
for cid, name, ctype, ctr0, cvr0, cpc0, budget, c_start, c_end, tw in CAMPAIGNS:
    for d, tm, promo in zip(DATES, TRAFFIC_MULT, PROMO):
        if not (c_start <= d <= c_end):
            continue
        w = tw
        if cid == "C03" and ANOMALY_START <= d <= ANOMALY_END:
            w = C03_BROKEN
        day_budget = budget * (1.6 if promo else 1.0) * rng.uniform(0.92, 1.05)
        fit_ctr = np.array([AGE_FIT[a] * TIER_FIT[t] for a, t in AUD])
        cell_ctr = ctr0 * fit_ctr * rng.lognormal(0, 0.08, len(AUD))
        cell_cvr = cvr0 * fit_ctr ** 1.5 * (1.25 if promo else 1.0) * rng.lognormal(0, 0.1, len(AUD))
        cell_cpc = cpc0 * (0.85 + 0.25 * fit_ctr) * rng.lognormal(0, 0.05, len(AUD))
        exp_cost_per_imp = (w * cell_ctr * cell_cpc).sum()
        total_imps = day_budget / exp_cost_per_imp * min(tm, 1.3) / 1.1
        imps = rng.poisson(total_imps * w)
        clicks = rng.binomial(imps, np.clip(cell_ctr, 0, 1))
        conv = rng.binomial(clicks, np.clip(cell_cvr, 0, 1))
        spend = np.round(clicks * cell_cpc, 2)
        aov = np.array([rng.lognormal(np.log(105), 0.15) for _ in AUD])
        gmv = np.round(conv * aov, 2)
        uv = np.round(clicks * rng.uniform(0.86, 0.93, len(AUD))).astype(int)
        if cid == "C02" and d == OUTAGE_DAY:
            clicks = np.zeros_like(clicks)
            uv = np.zeros_like(uv)
            conv = np.zeros_like(conv)
            gmv = np.zeros_like(gmv)
        for j, (a, t) in enumerate(AUD):
            ad_rows.append((d.date(), cid, a, t, imps[j], clicks[j], uv[j], conv[j], spend[j], gmv[j]))

ad_perf = pd.DataFrame(ad_rows, columns=[
    "date", "campaign_id", "age_group", "city_tier", "impressions", "clicks",
    "unique_visitors", "orders", "spend", "gmv"])

campaigns = pd.DataFrame([{
    "campaign_id": c[0], "campaign_name": c[1], "channel": c[2],
    "start_date": c[7].date(), "end_date": c[8].date(), "daily_budget": c[6],
} for c in CAMPAIGNS])


# ---------------------------------------------------------------- data-quality noise
# a few duplicated behaviour rows (logging retries), as raw exports often contain
dup = behaviour.sample(frac=0.002, random_state=SEED)
behaviour = pd.concat([behaviour, dup]).sort_values(["date", "customer_id"]).reset_index(drop=True)

# ---------------------------------------------------------------- write
customers.to_csv(OUT / "customers.csv", index=False)
behaviour.to_csv(OUT / "daily_user_behavior.csv", index=False)
orders.to_csv(OUT / "orders.csv", index=False)
campaigns.to_csv(OUT / "campaigns.csv", index=False)
ad_perf.to_csv(OUT / "ad_performance_daily.csv", index=False)
coupon_program.to_csv(OUT / "coupon_program_assignment.csv", index=False)

# answer key kept out of the analysis folder; used only to validate clustering
pd.DataFrame({"customer_id": customer_ids, "true_segment": seg}).to_csv(
    OUT.parent / "_answer_key_segments.csv", index=False)

for f in sorted(OUT.glob("*.csv")):
    print(f"{f.name:32s} {sum(1 for _ in open(f)) - 1:>9,d} rows")
