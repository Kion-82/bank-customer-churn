-- 1. Rank each customer's balance within their own country
SELECT
    c.customer_id,
    c.geography,
    a.balance,
    RANK() OVER (PARTITION BY c.geography ORDER BY a.balance DESC) AS balance_rank_in_country
FROM customers c
JOIN accounts a ON a.customer_id = c.customer_id
ORDER BY c.geography, balance_rank_in_country
LIMIT 20;

-- 2. Split customers into tenure quartiles within each country
SELECT
    c.customer_id,
    c.geography,
    c.tenure,
    NTILE(4) OVER (PARTITION BY c.geography ORDER BY c.tenure DESC) AS tenure_quartile
FROM customers c
LIMIT 20;

-- 3. Trailing moving average of estimated salary, ordered by age
SELECT
    customer_id,
    age,
    estimated_salary,
    ROUND(AVG(estimated_salary) OVER (
        ORDER BY age
        ROWS BETWEEN 100 PRECEDING AND CURRENT ROW
    ), 2) AS trailing_100_avg_salary_by_age
FROM customers
ORDER BY age
LIMIT 20;

-- 4. Cumulative churn rate as tenure increases
SELECT
    c.tenure,
    COUNT(*) AS n_customers,
    SUM(cs.exited) AS n_churned,
    ROUND(100.0 * SUM(SUM(cs.exited)) OVER (ORDER BY c.tenure)
          / SUM(COUNT(*)) OVER (ORDER BY c.tenure), 2) AS cumulative_churn_rate_pct
FROM customers c
JOIN churn_status cs ON cs.customer_id = c.customer_id
GROUP BY c.tenure
ORDER BY c.tenure;
