SELECT
    c.customer_id,
    c.credit_score,
    c.geography,
    c.gender,
    c.age,
    c.tenure,
    c.has_cr_card,
    c.is_active_member,
    c.estimated_salary,
    a.balance,
    a.num_of_products,
    a.card_type,
    a.point_earned,
    a.satisfaction_score,
    cs.exited,
    cs.complain
FROM customers c
JOIN accounts a      ON a.customer_id = c.customer_id
JOIN churn_status cs ON cs.customer_id = c.customer_id;