-- Daily paid-campaign KPIs with a CTR-drop alert.
-- A day is flagged when CTR is
--   * at least {{min_drop}}% below the campaign's trailing 14-day median, AND
--   * more than {{z}} robust standard deviations (1.4826 x MAD) below it, AND
--   * the campaign has at least 7 days of history.
-- Median / MAD are used instead of mean / std so promo spikes don't distort the baseline.

WITH daily AS (
    SELECT date, campaign_id,
           SUM(impressions) AS impressions,
           SUM(clicks)      AS clicks,
           SUM(orders)      AS orders,
           SUM(spend)       AS spend,
           SUM(gmv)         AS gmv
    FROM ad_performance
    GROUP BY 1, 2
),
kpi AS (
    SELECT *,
           clicks * 1.0 / NULLIF(impressions, 0) AS ctr,
           orders * 1.0 / NULLIF(clicks, 0)      AS cvr,
           gmv / NULLIF(spend, 0)                AS roas
    FROM daily
),
baseline AS (
    SELECT k.*,
           (SELECT MEDIAN(h.ctr) FROM kpi h
             WHERE h.campaign_id = k.campaign_id
               AND h.date BETWEEN k.date - INTERVAL 14 DAY AND k.date - INTERVAL 1 DAY) AS ctr_median_14d,
           (SELECT COUNT(*) FROM kpi h
             WHERE h.campaign_id = k.campaign_id
               AND h.date BETWEEN k.date - INTERVAL 14 DAY AND k.date - INTERVAL 1 DAY) AS history_days
    FROM kpi k
),
spread AS (
    SELECT b.*,
           (SELECT MEDIAN(ABS(h.ctr - b.ctr_median_14d)) FROM kpi h
             WHERE h.campaign_id = b.campaign_id
               AND h.date BETWEEN b.date - INTERVAL 14 DAY AND b.date - INTERVAL 1 DAY) AS ctr_mad_14d
    FROM baseline b
)
SELECT date, campaign_id, impressions, clicks, orders,
       ROUND(spend, 2) AS spend, ROUND(gmv, 2) AS gmv,
       ROUND(ctr * 100, 3) AS ctr_pct,
       ROUND(cvr * 100, 2) AS cvr_pct,
       ROUND(roas, 2)      AS roas,
       ROUND(ctr_median_14d * 100, 3) AS ctr_median_14d_pct,
       ROUND((ctr / ctr_median_14d - 1) * 100, 1) AS ctr_vs_median_pct,
       ROUND((ctr - ctr_median_14d) / NULLIF(1.4826 * ctr_mad_14d, 0), 1) AS ctr_robust_z,
       COALESCE(history_days >= 7
                AND ctr / ctr_median_14d - 1 <= -{{min_drop}} / 100.0
                AND (ctr - ctr_median_14d) / NULLIF(1.4826 * ctr_mad_14d, 0) < -{{z}}, FALSE) AS is_anomaly
FROM spread
ORDER BY campaign_id, date;
