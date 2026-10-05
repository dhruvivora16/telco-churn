-- Business questions to run against data/churn.db (e.g. in DB Browser for SQLite).
-- 1. Churn rate by contract type
SELECT Contract, COUNT(*) AS customers,
       ROUND(100.0 * AVG(Churn), 1) AS churn_pct
FROM clean_customers GROUP BY Contract ORDER BY churn_pct DESC;

-- 2. Churn by tenure bucket
SELECT CASE WHEN tenure < 12 THEN '0-11'
            WHEN tenure < 24 THEN '12-23'
            WHEN tenure < 48 THEN '24-47'
            ELSE '48+' END AS tenure_bucket,
       COUNT(*) AS customers, ROUND(100.0 * AVG(Churn), 1) AS churn_pct
FROM clean_customers GROUP BY tenure_bucket ORDER BY MIN(tenure);

-- 3. Monthly revenue lost to churned customers
SELECT ROUND(SUM(MonthlyCharges), 0) AS monthly_revenue_lost
FROM clean_customers WHERE Churn = 1;

-- 4. Churn by internet service and payment method
SELECT InternetService, PaymentMethod, COUNT(*) AS customers,
       ROUND(100.0 * AVG(Churn), 1) AS churn_pct
FROM clean_customers GROUP BY 1, 2 HAVING customers > 50 ORDER BY churn_pct DESC;
