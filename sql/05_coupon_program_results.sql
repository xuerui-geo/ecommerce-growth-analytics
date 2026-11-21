-- Coupon program readout: per-customer orders, buyer rate and net GMV for the
-- treatment and holdout (control) groups, before and after the 2023-09-01 launch.
-- Both windows are 91 days long.

WITH customer_period AS (
    SELECT cp.customer_id, cp."group", cp.segment_at_launch,
           CASE WHEN o.order_date BETWEEN DATE '2023-06-01' AND DATE '2023-08-30' THEN 'pre'
                WHEN o.order_date BETWEEN DATE '2023-09-01' AND DATE '2023-11-30' THEN 'post'
           END AS period,
           o.order_id, o.net_amount, o.discount_amount
    FROM coupon_program cp
    LEFT JOIN orders o
           ON o.customer_id = cp.customer_id AND o.order_status = 'completed'
),
per_customer AS (
    SELECT customer_id, "group", segment_at_launch,
           COUNT(order_id) FILTER (WHERE period = 'pre')          AS orders_pre,
           COUNT(order_id) FILTER (WHERE period = 'post')         AS orders_post,
           COALESCE(SUM(net_amount) FILTER (WHERE period = 'pre'), 0)       AS gmv_pre,
           COALESCE(SUM(net_amount) FILTER (WHERE period = 'post'), 0)      AS gmv_post,
           COALESCE(SUM(discount_amount) FILTER (WHERE period = 'post'), 0) AS discount_post
    FROM customer_period
    GROUP BY 1, 2, 3
)
SELECT * FROM per_customer ORDER BY customer_id;
