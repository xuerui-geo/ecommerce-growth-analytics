-- Weekly store KPI report: UV, PV, buyers, orders, GMV, conversion rate, AOV,
-- with week-over-week change. This is the recurring report sent to the business.

WITH traffic AS (
    SELECT DATE_TRUNC('week', date) AS week,
           COUNT(DISTINCT customer_id)  AS uv,
           SUM(page_views)              AS pv,
           COUNT(*)                     AS visits,
           SUM(add_to_cart)             AS add_to_cart
    FROM behavior
    GROUP BY 1
),
sales AS (
    SELECT DATE_TRUNC('week', order_date) AS week,
           COUNT(*)                       AS orders,
           COUNT(DISTINCT customer_id)    AS buyers,
           SUM(net_amount)                AS gmv,
           AVG(coupon_used)               AS coupon_rate
    FROM orders
    WHERE order_status = 'completed'
    GROUP BY 1
),
weekly AS (
    SELECT t.week, t.uv, t.pv, t.visits, t.add_to_cart,
           s.orders, s.buyers, ROUND(s.gmv, 0) AS gmv,
           ROUND(s.buyers * 100.0 / t.uv, 2)   AS conversion_rate_pct,
           ROUND(s.gmv / s.orders, 1)          AS aov,
           ROUND(s.coupon_rate * 100, 1)       AS coupon_rate_pct
    FROM traffic t JOIN sales s USING (week)
    -- keep complete weeks only (data runs Wed 2023-02-01 to Thu 2023-11-30)
    WHERE t.week >= DATE '2023-02-06' AND t.week <= DATE '2023-11-20'
)
SELECT *,
       ROUND((uv  / LAG(uv)  OVER (ORDER BY week) - 1) * 100, 1) AS uv_wow_pct,
       ROUND((gmv / LAG(gmv) OVER (ORDER BY week) - 1) * 100, 1) AS gmv_wow_pct
FROM weekly
ORDER BY week;
