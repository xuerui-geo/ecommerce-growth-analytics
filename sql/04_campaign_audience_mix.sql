-- Root-cause drill-down for one campaign: compare the audience mix and the
-- within-audience CTR / CVR between a baseline window and the anomaly window.

WITH periods AS (
    SELECT *,
           CASE WHEN date BETWEEN DATE '{{base_start}}' AND DATE '{{base_end}}' THEN 'baseline'
                WHEN date BETWEEN DATE '{{anom_start}}' AND DATE '{{anom_end}}' THEN 'anomaly'
           END AS period
    FROM ad_performance
    WHERE campaign_id = '{{campaign}}'
),
agg AS (
    SELECT period, age_group, city_tier,
           SUM(impressions) AS impressions, SUM(clicks) AS clicks,
           SUM(orders) AS orders, SUM(spend) AS spend, SUM(gmv) AS gmv
    FROM periods
    WHERE period IS NOT NULL
    GROUP BY 1, 2, 3
)
SELECT period, age_group, city_tier, impressions, clicks, orders,
       ROUND(spend, 2) AS spend, ROUND(gmv, 2) AS gmv,
       ROUND(impressions * 100.0 / SUM(impressions) OVER (PARTITION BY period), 2) AS impression_share_pct,
       ROUND(clicks * 100.0 / NULLIF(impressions, 0), 3) AS ctr_pct,
       ROUND(orders * 100.0 / NULLIF(clicks, 0), 2)      AS cvr_pct
FROM agg
ORDER BY period, age_group, city_tier;
