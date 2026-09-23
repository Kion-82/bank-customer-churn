-- Churn rate by country
SELECT
    c.geography,
    COUNT(*) AS n_customers,
    SUM(cs.exited) AS n_churned,
    ROUND(100.0 * SUM(cs.exited) / COUNT(*), 2) AS churn_rate_pct
FROM customers c
JOIN churn_status cs ON cs.customer_id = c.customer_id
GROUP BY c.geography
ORDER BY churn_rate_pct DESC;

-- Churn rate by number of products held
SELECT
    a.num_of_products,
    COUNT(*) AS n_customers,
    SUM(cs.exited) AS n_churned,
    ROUND(100.0 * SUM(cs.exited) / COUNT(*), 2) AS churn_rate_pct
FROM accounts a
JOIN churn_status cs ON cs.customer_id = a.customer_id
GROUP BY a.num_of_products
ORDER BY a.num_of_products;