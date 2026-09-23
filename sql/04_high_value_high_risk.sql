WITH stats AS (
    SELECT
        (SELECT AVG(balance) FROM accounts)           AS avg_balance,
        (SELECT AVG(estimated_salary) FROM customers) AS avg_salary
),
segmented AS (
    SELECT
        cs.exited,
        CASE
            WHEN a.balance >= s.avg_balance AND c.estimated_salary >= s.avg_salary
                THEN 'high_value'
            ELSE 'standard_value'
        END AS value_segment,
        CASE
            WHEN c.is_active_member = 0 OR a.num_of_products = 1
                THEN 'high_risk'
            ELSE 'lower_risk'
        END AS risk_segment
    FROM customers c
    JOIN accounts a      ON a.customer_id = c.customer_id
    JOIN churn_status cs ON cs.customer_id = c.customer_id
    CROSS JOIN stats s
)
SELECT
    value_segment,
    risk_segment,
    COUNT(*) AS n_customers,
    SUM(exited) AS n_churned,
    ROUND(100.0 * SUM(exited) / COUNT(*), 2) AS churn_rate_pct
FROM segmented
GROUP BY value_segment, risk_segment
ORDER BY churn_rate_pct DESC;