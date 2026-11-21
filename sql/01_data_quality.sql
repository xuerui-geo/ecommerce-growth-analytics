-- Data-quality checks run before any analysis.
-- Returns one row per check with the number of offending records.

SELECT 'behavior: exact duplicate rows' AS check_name,
       (SELECT COUNT(*) FROM behavior_raw) - (SELECT COUNT(*) FROM behavior) AS issues
UNION ALL
SELECT 'customers: missing gender', COUNT(*) FROM customers WHERE gender IS NULL
UNION ALL
SELECT 'orders: refunded (excluded from revenue KPIs)', COUNT(*) FROM orders WHERE order_status = 'refunded'
UNION ALL
SELECT 'orders: net_amount <= 0', COUNT(*) FROM orders WHERE net_amount <= 0
UNION ALL
SELECT 'orders: customer not in customers table', COUNT(*)
FROM orders o LEFT JOIN customers c USING (customer_id) WHERE c.customer_id IS NULL
UNION ALL
-- tracking outage: a campaign spent money and served impressions but logged zero clicks
SELECT 'ads: campaign-days with spend > 0 but 0 clicks', COUNT(*)
FROM (
    SELECT date, campaign_id
    FROM ad_performance
    GROUP BY 1, 2
    HAVING SUM(spend) > 0 AND SUM(clicks) = 0
);
