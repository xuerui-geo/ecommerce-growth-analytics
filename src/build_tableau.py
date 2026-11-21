"""
Build the Tableau dashboard for the project.

1. Exports tidy, pre-aggregated CSVs to tableau/Data/tableau/ (from the same SQL / pipeline as the notebooks).
2. Writes tableau/ecommerce_growth_dashboard.twb (connects to those CSVs with relative paths).
3. Packages tableau/ecommerce_growth_dashboard.twbx (workbook + data, opens directly).

Usage:  python src/build_tableau.py
"""

import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402
import segmentation as seg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "tableau"
DATA_REL = "Data/tableau"
DATA = TAB / DATA_REL
DATA.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#a3a29c"

# =====================================================================  1. data
con = db.connect()

weekly = db.run(con, "02_weekly_store_kpis")[["week", "uv", "pv", "visits", "orders", "buyers", "gmv"]]
weekly["week"] = pd.to_datetime(weekly["week"]).dt.strftime("%Y-%m-%d")
weekly.to_csv(DATA / "weekly_kpis.csv", index=False)

camp = con.execute("""
    SELECT c.campaign_id || ' ' || c.campaign_name AS campaign, c.channel,
           SUM(impressions) AS impressions, SUM(clicks) AS clicks, SUM(a.orders) AS orders,
           ROUND(SUM(spend), 2) AS spend, ROUND(SUM(gmv), 2) AS gmv
    FROM ad_performance a JOIN campaigns c USING (campaign_id)
    GROUP BY 1, 2 ORDER BY 1
""").df()
camp.to_csv(DATA / "campaign_summary.csv", index=False)

daily = db.run(con, "03_campaign_daily_anomalies", z=4, min_drop=20)
c03 = daily[daily["campaign_id"] == "C03"].copy()
c03["date"] = pd.to_datetime(c03["date"]).dt.strftime("%Y-%m-%d")
c03["alert"] = np.where(c03["is_anomaly"], "CTR alert", "Normal")
c03 = c03[["date", "impressions", "clicks", "orders", "spend", "gmv", "alert"]]
c03.to_csv(DATA / "c03_daily.csv", index=False)

mix = db.run(con, "04_campaign_audience_mix", campaign="C03", base_start="2023-06-19",
             base_end="2023-07-11", anom_start="2023-07-12", anom_end="2023-07-30")
mix["period"] = mix["period"].map({"baseline": "Before", "anomaly": "Incident"})
for col, fname in [("age_group", "c03_mix_age.csv"), ("city_tier", "c03_mix_tier.csv")]:
    t = mix.groupby(["period", col])["impressions"].sum()
    sh = (t / t.groupby(level=0).transform("sum")).round(4).rename("impression_share").reset_index()
    sh.rename(columns={col: "audience"})[["audience", "period", "impression_share"]].to_csv(DATA / fname, index=False)
(DATA / "c03_audience_mix.csv").unlink(missing_ok=True)

share = mix.pivot_table(index=["age_group", "city_tier"], columns="period", values="impression_share_pct") / 100
ctr = mix.pivot_table(index=["age_group", "city_tier"], columns="period", values="ctr_pct")
b, a = "Before", "Incident"
decomp = pd.DataFrame({
    "component": ["1 Baseline CTR", "2 Audience-mix effect", "3 Within-audience effect", "4 Interaction",
                  "5 Incident CTR"],
    "ctr_points": [(share[b] * ctr[b]).sum(), ((share[a] - share[b]) * ctr[b]).sum(),
                   (share[b] * (ctr[a] - ctr[b])).sum(), ((share[a] - share[b]) * (ctr[a] - ctr[b])).sum(),
                   (share[a] * ctr[a]).sum()],
})
decomp["ctr_points"] = decomp["ctr_points"].round(3)
decomp.to_csv(DATA / "ctr_decomposition.csv", index=False)

behavior = con.execute("SELECT * FROM behavior").df()
orders = con.execute("SELECT * FROM orders").df()
customers = con.execute("SELECT * FROM customers").df()
feats = seg.build_features(behavior, orders, customers, "2023-08-31")
labels, _, _ = seg.fit_segments(feats)
feats["segment"] = labels.map(seg.name_clusters(feats, labels))
gmv = (orders[(orders.order_status == "completed") & (pd.to_datetime(orders.order_date) <= "2023-08-31")]
       .groupby("customer_id")["net_amount"].sum())
feats["gmv"] = gmv.reindex(feats.index).fillna(0)
s = feats.groupby("segment").agg(customers=("gmv", "size"), gmv=("gmv", "sum"))
seg_long = pd.concat([
    pd.DataFrame({"segment": s.index, "metric": "% of customers", "share": (s.customers / s.customers.sum()).round(4)}),
    pd.DataFrame({"segment": s.index, "metric": "% of GMV (Feb-Aug)", "share": (s.gmv / s.gmv.sum()).round(4)}),
])
seg_long.to_csv(DATA / "segment_share.csv", index=False)

prof = feats.groupby("segment").agg(
    customers=("gmv", "size"), avg_order_value=("avg_order_value", "median"),
    orders_per_30d=("orders_per_30d", "mean"), coupon_share=("coupon_share", "mean"),
    recency_days=("recency_days", "median"), tenure_days=("tenure_days", "median")).round(3).reset_index()
prof["action"] = prof["segment"].map({
    "High-Value Loyal": "No coupons - VIP perks", "Deal Seekers": "TARGET: coupons",
    "New Potential": "TARGET: first-order coupon", "Window Shoppers": "Fix product pages first",
    "Dormant / At-Risk": "Low-cost win-back"})
prof.to_csv(DATA / "segment_profile.csv", index=False)

wk = con.execute("""
    SELECT DATE_TRUNC('week', o.order_date) AS week, cp."group" AS grp, COUNT(*) AS orders
    FROM orders o JOIN coupon_program cp USING (customer_id)
    WHERE o.order_status = 'completed' AND o.order_date BETWEEN DATE '2023-06-05' AND DATE '2023-11-26'
    GROUP BY 1, 2 ORDER BY 1, 2
""").df()
sizes = con.execute('SELECT "group", COUNT(*) n FROM coupon_program GROUP BY 1').df().set_index("group")["n"]
wk["group"] = wk["grp"].map({"treatment": "Treatment (coupons)", "control": "Control (holdout)"})
wk["customers"] = wk["grp"].map(sizes)
wk["week"] = pd.to_datetime(wk["week"]).dt.strftime("%Y-%m-%d")
wk[["week", "group", "orders", "customers"]].to_csv(DATA / "experiment_weekly.csv", index=False)

pc = db.run(con, "05_coupon_program_results")
exp = pc.groupby("group").agg(customers=("customer_id", "size"), orders_post=("orders_post", "sum"),
                              gmv_post=("gmv_post", "sum")).reset_index()
exp["group"] = exp["group"].map({"treatment": "Treatment (coupons)", "control": "Control (holdout)"})
exp.to_csv(DATA / "experiment_summary.csv", index=False)

# =====================================================================  2. Hyper extracts
# Tableau Public only opens workbooks whose data sources are extracts, so every CSV is
# also written as a .hyper file (schema "Extract", table "Extract") and packaged with the workbook.
from tableauhyperapi import (Connection, CreateMode, HyperProcess, Inserter, SqlType, TableDefinition,  # noqa: E402
                             TableName, Telemetry)

EXTRACTS_REL = "Data/Extracts"
EXTRACTS = TAB / EXTRACTS_REL
EXTRACTS.mkdir(parents=True, exist_ok=True)


def write_hyper(csv_name, types):
    df = pd.read_csv(DATA / csv_name)
    sql = {"date": SqlType.date(), "integer": SqlType.big_int(), "real": SqlType.double(), "string": SqlType.text()}
    table = TableDefinition(TableName("Extract", "Extract"),
                            [TableDefinition.Column(c, sql[t]) for c, t in types.items()])
    path = EXTRACTS / csv_name.replace(".csv", ".hyper")
    with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU,
                      parameters={"default_database_version": "0"}) as hp:
        with Connection(hp.endpoint, path, CreateMode.CREATE_AND_REPLACE) as con_h:
            con_h.catalog.create_schema("Extract")
            con_h.catalog.create_table(table)
            rows = []
            for rec in df.itertuples(index=False):
                row = []
                for (c, t), v in zip(types.items(), rec):
                    if pd.isna(v):
                        row.append(None)
                    elif t == "date":
                        row.append(pd.Timestamp(v).date())
                    elif t == "integer":
                        row.append(int(v))
                    elif t == "real":
                        row.append(float(v))
                    else:
                        row.append(str(v))
                rows.append(row)
            with Inserter(con_h, table) as ins:
                ins.add_rows(rows)
                ins.execute()
    return path


# =====================================================================  3. workbook XML
REMOTE = {"date": 133, "string": 129, "integer": 20, "real": 5}


def infer(series):
    if series.name in ("week", "date"):
        return "date"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "real"
    return "string"


class DS:
    """A text-file datasource plus calculated fields."""

    def __init__(self, key, file, caption, calcs=None, colors=None, formats=None):
        self.key, self.file, self.caption = key, file, caption
        self.df = pd.read_csv(DATA / file)
        self.types = {c: infer(self.df[c]) for c in self.df.columns}
        self.calcs = calcs or {}      # name -> (caption, formula, format)
        self.colors = colors or {}    # field -> {value: hex}
        self.formats = formats or {}  # field -> Tableau number format
        self.name = f"federated.{key}"

    def ref(self, inst):
        return f"[{self.name}].[{inst}]"

    def xml(self):
        table = self.file.replace(".csv", "#csv")
        cols = "\n".join(
            f"              <column datatype='{t}' name='{escape(c)}' ordinal='{i}' />"
            for i, (c, t) in enumerate(self.types.items()))
        meta = "\n".join(f"""          <metadata-record class='column'>
            <remote-name>{escape(c)}</remote-name>
            <remote-type>{REMOTE[t]}</remote-type>
            <local-name>[{escape(c)}]</local-name>
            <parent-name>[{self.file}]</parent-name>
            <remote-alias>{escape(c)}</remote-alias>
            <ordinal>{i}</ordinal>
            <local-type>{t}</local-type>
            <aggregation>{'Sum' if t in ('integer', 'real') else ('Year' if t == 'date' else 'Count')}</aggregation>
            <contains-null>true</contains-null>
          </metadata-record>""" for i, (c, t) in enumerate(self.types.items()))
        fields = []
        for c, t in self.types.items():
            if t in ("integer", "real"):
                fa = f" default-format='{self.formats[c]}'" if c in self.formats else ""
                fields.append(f"      <column datatype='{t}'{fa} name='[{c}]' role='measure' type='quantitative' />")
            elif t == "date":
                fields.append(f"      <column datatype='date' name='[{c}]' role='dimension' type='ordinal' />")
            else:
                fields.append(f"      <column datatype='string' name='[{c}]' role='dimension' type='nominal' />")
        for name, (cap, formula, fmt) in self.calcs.items():
            fmt_attr = f" default-format='{fmt}'" if fmt else ""
            fields.append(
                f"      <column caption='{escape(cap)}' datatype='real'{fmt_attr} name='[{name}]' role='measure' "
                f"type='quantitative'>\n        <calculation class='tableau' formula='{escape(formula, {chr(39): '&apos;'})}' />\n"
                f"      </column>")
        style = ""
        if self.colors:
            rules = []
            for field, mapping in self.colors.items():
                maps = "\n".join(
                    f"          <map to='{hx}'>\n            <bucket>&quot;{escape(v)}&quot;</bucket>\n          </map>"
                    for v, hx in mapping.items())
                rules.append(f"        <encoding attr='color' field='[none:{field}:nk]' type='palette'>\n{maps}\n        </encoding>")
            style = "    <style>\n      <style-rule element='mark'>\n" + "\n".join(rules) + "\n      </style-rule>\n    </style>\n"
        hyper = self.file.replace(".csv", ".hyper")
        ex_meta = "\n".join(f"""            <metadata-record class='column'>
              <remote-name>{escape(c)}</remote-name>
              <remote-type>{REMOTE[t]}</remote-type>
              <local-name>[{escape(c)}]</local-name>
              <parent-name>[Extract]</parent-name>
              <remote-alias>{escape(c)}</remote-alias>
              <ordinal>{i}</ordinal>
              <local-type>{t}</local-type>
              <aggregation>{'Sum' if t in ('integer', 'real') else ('Year' if t == 'date' else 'Count')}</aggregation>
              <contains-null>true</contains-null>
            </metadata-record>""" for i, (c, t) in enumerate(self.types.items()))
        extract = f"""      <extract count='-1' enabled='true' units='records'>
        <connection access_mode='readonly' authentication='auth-none' author-locale='en_US' class='hyper' dbname='{EXTRACTS_REL}/{hyper}' default-settings='yes' schema='Extract' sslmode='' tablename='Extract' update-time='11/19/2026 08:00:00 PM' username='tableau_internal_user'>
          <relation name='Extract' table='[Extract].[Extract]' type='table' />
          <refresh increment-key='' incremental-updates='false' />
          <metadata-records>
{ex_meta}
          </metadata-records>
        </connection>
      </extract>
"""
        return f"""    <datasource caption='{escape(self.caption)}' inline='true' name='{self.name}' version='18.1'>
      <connection class='federated'>
        <named-connections>
          <named-connection caption='{self.file}' name='textscan.{self.key}'>
            <connection class='textscan' directory='{DATA_REL}' filename='{self.file}' password='' server='' />
          </named-connection>
        </named-connections>
        <relation connection='textscan.{self.key}' name='{self.file}' table='[{table}]' type='table'>
          <columns character-set='UTF-8' header='yes' locale='en_US' separator=','>
{cols}
          </columns>
        </relation>
        <metadata-records>
{meta}
        </metadata-records>
      </connection>
      <aliases enabled='yes' />
{chr(10).join(fields)}
{extract}      <layout dim-ordering='alphabetic' dim-percentage='0.5' measure-ordering='alphabetic' measure-percentage='0.4' show-structure='true' />
{style}    </datasource>"""


def inst(ds, spec):
    """spec like 'sum:gmv', 'none:segment', 'tdy:week', 'usr:Calculation_1' -> (instance name, xml)."""
    kind, field = spec.split(":", 1)
    if kind == "none":
        name, deriv, typ = f"none:{field}:nk", "None", "nominal"
    elif kind == "tdy":
        name, deriv, typ = f"tdy:{field}:qk", "Day-Trunc", "quantitative"
    elif kind == "usr":
        name, deriv, typ = f"usr:{field}:qk", "User", "quantitative"
    else:
        name, deriv, typ = f"{kind}:{field}:qk", {"sum": "Sum", "avg": "Avg"}[kind], "quantitative"
    xml = f"            <column-instance column='[{field}]' derivation='{deriv}' name='[{name}]' pivot='key' type='{typ}' />"
    return name, xml


def worksheet(title, ds, cols, rows, mark, color=None, label=None, sort_desc=None, label_fmt=None):
    specs = [s for s in cols + rows + [color, label] if s]
    deps, seen, insts = [], set(), {}
    for s in specs:
        n, x = inst(ds, s)
        insts[s] = n
        if n not in seen:
            deps.append(x)
            seen.add(n)
    fields_needed = {s.split(":", 1)[1] for s in specs}
    # base columns referenced inside calculated fields must also be declared
    for f in list(fields_needed):
        if f in ds.calcs:
            fields_needed |= set(re.findall(r"\[([^\]]+)\]", ds.calcs[f][1]))
    fields_needed = sorted(fields_needed)
    col_decl = []
    for f in fields_needed:
        if f in ds.calcs:
            cap, formula, fmt = ds.calcs[f]
            fmt_attr = f" default-format='{fmt}'" if fmt else ""
            col_decl.append(
                f"            <column caption='{escape(cap)}' datatype='real'{fmt_attr} name='[{f}]' role='measure' "
                f"type='quantitative'>\n              <calculation class='tableau' formula='{escape(formula, {chr(39): '&apos;'})}' />\n"
                f"            </column>")
        else:
            t = ds.types[f]
            role, typ = (("measure", "quantitative") if t in ("integer", "real")
                         else ("dimension", "ordinal" if t == "date" else "nominal"))
            fa = f" default-format='{ds.formats[f]}'" if f in ds.formats else ""
            col_decl.append(f"            <column datatype='{t}'{fa} name='[{f}]' role='{role}' type='{typ}' />")
    enc = []
    if color:
        enc.append(f"              <color column='{ds.ref(insts[color])}' />")
    if label:
        enc.append(f"              <text column='{ds.ref(insts[label])}' />")
    enc_xml = ("            <encodings>\n" + "\n".join(enc) + "\n            </encodings>\n") if enc else ""
    pane_style = ""
    if label:
        pane_style = ("            <style>\n              <style-rule element='mark'>\n"
                      "                <format attr='mark-labels-show' value='true' />\n"
                      "              </style-rule>\n            </style>\n")
    sort_xml = ""
    if sort_desc:
        dim, meas = sort_desc
        sort_xml = (f"          <sort class='computed' column='{ds.ref(insts[dim])}' direction='DESC' "
                    f"using='{ds.ref(insts[meas])}' />\n")

    def shelf(lst):
        refs = [ds.ref(insts[s]) for s in lst]
        if not refs:
            return ""
        out = refs[0]
        for r in refs[1:]:
            out = f"({out} / {r})"
        return out

    return f"""    <worksheet name='{escape(title)}'>
      <layout-options>
        <title>
          <formatted-text>
            <run bold='true' fontsize='12'>{escape(title)}</run>
          </formatted-text>
        </title>
      </layout-options>
      <table>
        <view>
          <datasources>
            <datasource caption='{escape(ds.caption)}' name='{ds.name}' />
          </datasources>
          <datasource-dependencies datasource='{ds.name}'>
{chr(10).join(col_decl)}
{chr(10).join(deps)}
          </datasource-dependencies>
{sort_xml}          <aggregation value='true' />
        </view>
        <style />
        <panes>
          <pane selection-relaxation-option='selection-relaxation-allow'>
            <view>
              <breakdown value='auto' />
            </view>
            <mark class='{mark}' />
{enc_xml}{pane_style}          </pane>
        </panes>
        <rows>{shelf(rows)}</rows>
        <cols>{shelf(cols)}</cols>
      </table>
    </worksheet>"""


def uuid_for(s):
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, s)).upper()


# ---------- datasources
ds_week = DS("weekly", "weekly_kpis.csv", "Weekly KPIs", calcs={
    "Calculation_1001": ("Conversion rate (buyers / UV)", "SUM([buyers]) / SUM([uv])", "p0.0%"),
    "Calculation_1002": ("AOV (CNY)", "SUM([gmv]) / SUM([orders])", "n#,##0"),
})
ds_camp = DS("campaigns", "campaign_summary.csv", "Campaign Summary", calcs={
    "Calculation_2001": ("ROAS", "SUM([gmv]) / SUM([spend])", "n#,##0.00"),
    "Calculation_2002": ("CTR", "SUM([clicks]) / SUM([impressions])", "p0.00%"),
}, colors={"channel": {"Paid Search": BLUE, "Display Ads": ORANGE, "Smart Campaign": AQUA,
                       "Livestream/Influencer": YELLOW}})
ds_c03 = DS("c03", "c03_daily.csv", "C03 Daily", calcs={
    "Calculation_3001": ("CTR", "SUM([clicks]) / SUM([impressions])", "p0.00%"),
    "Calculation_3002": ("ROAS", "SUM([gmv]) / SUM([spend])", "n#,##0.00"),
}, colors={"alert": {"Normal": BLUE, "CTR alert": ORANGE}})
period_colors = {"Before": BLUE, "Incident": ORANGE}
ds_mix_age = DS("mixage", "c03_mix_age.csv", "C03 Mix by Age", colors={"period": period_colors},
                formats={"impression_share": "p0%"})
ds_mix_tier = DS("mixtier", "c03_mix_tier.csv", "C03 Mix by City Tier", colors={"period": period_colors},
                 formats={"impression_share": "p0%"})
ds_dec = DS("decomp", "ctr_decomposition.csv", "CTR Decomposition")
ds_seg = DS("segshare", "segment_share.csv", "Segment Share", formats={"share": "p0%"},
            colors={"metric": {"% of customers": GRAY, "% of GMV (Feb-Aug)": BLUE}})
ds_prof = DS("segprofile", "segment_profile.csv", "Segment Profile")
grp_colors = {"Treatment (coupons)": BLUE, "Control (holdout)": GRAY}
ds_expw = DS("expweekly", "experiment_weekly.csv", "Experiment Weekly", calcs={
    "Calculation_4001": ("Orders per 100 customers", "SUM([orders]) / SUM([customers]) * 100", "n#,##0.0"),
}, colors={"group": grp_colors})
ds_exp = DS("expsummary", "experiment_summary.csv", "Experiment Summary", calcs={
    "Calculation_5001": ("Orders per customer (Sep-Nov)", "SUM([orders_post]) / SUM([customers])", "n#,##0.00"),
    "Calculation_5002": ("Net GMV per customer (Sep-Nov)", "SUM([gmv_post]) / SUM([customers])", "n¥#,##0"),
}, colors={"group": grp_colors})
ALL_DS = [ds_week, ds_camp, ds_c03, ds_mix_age, ds_mix_tier, ds_dec, ds_seg, ds_prof, ds_expw, ds_exp]
for _d in ALL_DS:
    write_hyper(_d.file, _d.types)

# ---------- worksheets
WS = {
    "Weekly GMV": worksheet("Weekly GMV (CNY)", ds_week, ["tdy:week"], ["sum:gmv"], "Line"),
    "Weekly UV": worksheet("Weekly unique visitors", ds_week, ["tdy:week"], ["sum:uv"], "Line"),
    "Weekly CVR": worksheet("Conversion rate", ds_week, ["tdy:week"], ["usr:Calculation_1001"], "Line"),
    "Weekly AOV": worksheet("Average order value", ds_week, ["tdy:week"], ["usr:Calculation_1002"], "Line"),
    "ROAS": worksheet("ROAS by campaign", ds_camp, ["usr:Calculation_2001"], ["none:campaign"], "Bar",
                      color="none:channel", label="usr:Calculation_2001",
                      sort_desc=("none:campaign", "usr:Calculation_2001")),
    "C03 CTR": worksheet("C03 Visitor Retargeting - daily CTR", ds_c03, ["tdy:date"], ["usr:Calculation_3001"],
                         "Circle", color="none:alert"),
    "C03 ROAS": worksheet("C03 Visitor Retargeting - daily ROAS", ds_c03, ["tdy:date"], ["usr:Calculation_3002"],
                          "Line"),
    "MixAge": worksheet("C03 impression share by age: before vs incident", ds_mix_age,
                        ["none:audience", "none:period"], ["sum:impression_share"], "Bar", color="none:period",
                        label="sum:impression_share"),
    "MixTier": worksheet("C03 impression share by city tier", ds_mix_tier,
                         ["none:audience", "none:period"], ["sum:impression_share"], "Bar", color="none:period",
                         label="sum:impression_share"),
    "Decomp": worksheet("What explains the CTR drop? (CTR points)", ds_dec, ["sum:ctr_points"], ["none:component"],
                        "Bar", label="sum:ctr_points"),
    "SegShare": worksheet("Segment size vs revenue share", ds_seg, ["sum:share"], ["none:segment", "none:metric"],
                          "Bar", color="none:metric", label="sum:share"),
    "SegProfile": worksheet("Segment profile & action", ds_prof, [],
                            ["none:segment", "none:action"], "Text", label="sum:customers"),
    "ExpWeekly": worksheet("Weekly orders per 100 targeted customers", ds_expw, ["tdy:week"],
                           ["usr:Calculation_4001"], "Line", color="none:group"),
    "ExpOrders": worksheet("Orders per customer, Sep-Nov (+49%)", ds_exp, ["none:group"],
                           ["usr:Calculation_5001"], "Bar", color="none:group", label="usr:Calculation_5001"),
    "ExpGMV": worksheet("Net GMV per customer, Sep-Nov", ds_exp, ["none:group"],
                        ["usr:Calculation_5002"], "Bar", color="none:group", label="usr:Calculation_5002"),
}
TITLES = {k: v.split("name='", 1)[1].split("'", 1)[0] for k, v in WS.items()}


# ---------- dashboards: standard tiled layout (layout-basic > vertical flow > horizontal flows)
_ids = iter(range(100, 10000))


def sheet_zone(key, x, y, w, h):
    return f"<zone h='{h}' id='{next(_ids)}' name='{escape(TITLES[key])}' w='{w}' x='{x}' y='{y}' />"


def flow(direction, children, x, y, w, h):
    return (f"<zone h='{h}' id='{next(_ids)}' param='{direction}' type-v2='layout-flow' w='{w}' x='{x}' y='{y}'>"
            + "".join(children) + "</zone>")


def title_zone(title, sub, x, y, w, h):
    return (f"<zone h='{h}' id='{next(_ids)}' type-v2='text' w='{w}' x='{x}' y='{y}'>"
            f"<formatted-text><run bold='true' fontsize='16'>{escape(title)}</run><run>Æ&#10;</run>"
            f"<run fontcolor='#52514e' fontsize='10'>{escape(sub)}</run></formatted-text></zone>")


def row(items, y, h):
    """items: list of (key, width) or (['k1', 'k2'], width) for a vertical stack."""
    kids, x = [], 0
    for item, w in items:
        if isinstance(item, list):
            hh = h // len(item)
            sub = [sheet_zone(k, x, y + i * hh, w, hh if i < len(item) - 1 else h - hh * (len(item) - 1))
                   for i, k in enumerate(item)]
            kids.append(flow("vert", sub, x, y, w, h))
        else:
            kids.append(sheet_zone(item, x, y, w, h))
        x += w
    return flow("horz", kids, 0, y, 100000, h)


def dashboard(name, title, sub, rows_spec, with_text=True):
    th = 9000
    body_h = (100000 - th) // len(rows_spec)
    parts = [title_zone(title, sub, 0, 0, 100000, th)]
    for i, items in enumerate(rows_spec):
        h = body_h if i < len(rows_spec) - 1 else 100000 - th - body_h * (len(rows_spec) - 1)
        parts.append(row(items, th + i * body_h, h))
    body = flow("vert", parts, 0, 0, 100000, 100000)
    root = f"<zone h='100000' id='{next(_ids)}' type-v2='layout-basic' w='100000' x='0' y='0'>{body}</zone>"
    return f"""    <dashboard name='{escape(name)}'>
      <style />
      <size maxheight='900' maxwidth='1400' minheight='900' minwidth='1400' />
      <zones>
        {root}
      </zones>
    </dashboard>"""


DASHBOARDS = [
    ("1 Store KPIs", "Store KPIs - Feb to Nov 2023",
     "Weekly traffic, conversion and revenue. Spikes = 618 (June) and Singles' Day (Oct 24-Nov 11) sales. Synthetic data.",
     [[("Weekly GMV", 50000), ("Weekly UV", 50000)], [("Weekly CVR", 50000), ("Weekly AOV", 50000)]]),
    ("2 Campaign Diagnostics", "Paid Campaign Diagnostics - the July C03 incident",
     "C03 targeting drifted to 18-24 / Tier 3-4+ on Jul 12-30 (Before = Jun 19-Jul 11): CTR 3.35% to 2.06%, ROAS 5.6 to 3.4, about CNY 47K GMV lost. 97% of the CTR drop is an audience-mix effect.",
     [[("ROAS", 40000), (["C03 CTR", "C03 ROAS"], 60000)], [("MixAge", 37000), ("MixTier", 33000), ("Decomp", 30000)]]),
    ("3 Segments and Coupon Test", "Customer Segments and Coupon Program A/B Test",
     "K-means found 5 segments; Deal Seekers + New Potential got coupons from Sep 1 with a random 20% holdout. Orders per customer +49% (95% CI +37% to +63%).",
     [[("SegShare", 50000), ("SegProfile", 50000)], [("ExpWeekly", 50000), ("ExpOrders", 25000), ("ExpGMV", 25000)]]),
]


def dash_keys(rows_spec):
    out = []
    for items in rows_spec:
        for item, _ in items:
            out += item if isinstance(item, list) else [item]
    return out


dash_xml = [dashboard(n, t, sb, z) for n, t, sb, z in DASHBOARDS]

windows = []
for n, _, _, z in DASHBOARDS:
    vps = "\n".join(f"        <viewpoint name='{escape(TITLES[k])}'>\n          <zoom type='entire-view' />\n        </viewpoint>"
                    for k in dash_keys(z))
    windows.append(f"    <window class='dashboard' maximized='true' name='{escape(n)}'>\n      <viewpoints>\n{vps}\n"
                   f"      </viewpoints>\n      <active id='-1' />\n    </window>")

CARDS = """      <cards>
        <edge name='left'>
          <strip size='160'>
            <card type='pages' />
            <card type='filters' />
            <card type='marks' />
          </strip>
        </edge>
        <edge name='top'>
          <strip size='2147483647'>
            <card type='columns' />
          </strip>
          <strip size='2147483647'>
            <card type='rows' />
          </strip>
          <strip size='31'>
            <card type='title' />
          </strip>
        </edge>
      </cards>"""


def ws_window(key, hidden=True):
    attr = "hidden='true'" if hidden else "maximized='true'"
    return f"    <window class='worksheet' {attr} name='{escape(TITLES[key])}'>\n{CARDS}\n    </window>"


def make_twb(ds_list, ws_list, dashes, wins):
    dash_block = f"  <dashboards>\n{chr(10).join(dashes)}\n  </dashboards>\n" if dashes else ""
    return f"""<?xml version='1.0' encoding='utf-8' ?>

<workbook original-version='18.1' source-build='2023.1.0 (20231.23.0310.1045)' source-platform='win' version='18.1' xmlns:user='http://www.tableausoftware.com/xml/user'>
  <preferences>
    <preference name='ui.encoding.shelf.height' value='24' />
    <preference name='ui.shelf.height' value='26' />
  </preferences>
  <datasources>
{chr(10).join(d.xml() for d in ds_list)}
  </datasources>
  <worksheets>
{chr(10).join(ws_list)}
  </worksheets>
{dash_block}  <windows source-height='30'>
{chr(10).join(wins)}
  </windows>
</workbook>
"""


def package(name, twb_text, csv_files):
    twb_path = TAB / f"{name}.twb"
    twb_path.write_text(twb_text, encoding="utf-8")
    twbx_path = TAB / f"{name}.twbx"
    with zipfile.ZipFile(twbx_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(twb_path, twb_path.name)
        for f in csv_files:
            z.write(DATA / f, f"{DATA_REL}/{f}")
            h = f.replace(".csv", ".hyper")
            z.write(EXTRACTS / h, f"{EXTRACTS_REL}/{h}")
    return twb_path, twbx_path


full = make_twb(ALL_DS, list(WS.values()), dash_xml, windows + [ws_window(k) for k in WS])
twb_path, twbx_path = package("ecommerce_growth_dashboard", full, sorted(f.name for f in DATA.glob("*.csv")))

# minimal smoke-test workbook: one data source, one line chart, no dashboard
if "--minimal" in sys.argv:
    mini = make_twb([ds_week], [WS["Weekly GMV"]], [], [ws_window("Weekly GMV", hidden=False)])
    package("test_minimal", mini, ["weekly_kpis.csv"])
    sheets = make_twb(ALL_DS, list(WS.values()), [], [ws_window(k, hidden=False) for k in WS])
    one = dashboard("Test", "Test dashboard", "one chart", [[("Weekly GMV", 100000)]])
    one_win = [f"    <window class='dashboard' maximized='true' name='Test'>\n      <viewpoints>\n"
               f"        <viewpoint name='{TITLES['Weekly GMV']}'>\n          <zoom type='entire-view' />\n"
               f"        </viewpoint>\n      </viewpoints>\n      <active id='-1' />\n    </window>",
               ws_window("Weekly GMV", hidden=False)]
    package("test_one_dashboard", make_twb([ds_week], [WS["Weekly GMV"]], [one], one_win), ["weekly_kpis.csv"])
    package("test_sheets_only", sheets, sorted(f.name for f in DATA.glob("*.csv")))

print("wrote", twb_path.relative_to(ROOT), "and", twbx_path.relative_to(ROOT))
for f in sorted(DATA.glob("*.csv")):
    print(f"  {f.name:28s} {len(pd.read_csv(f)):>5} rows")
