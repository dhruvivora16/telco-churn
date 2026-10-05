-- Cleans the raw Telco table loaded by src/data_prep.py (SQLite dialect).
DROP TABLE IF EXISTS clean_customers;

CREATE TABLE clean_customers AS
SELECT
    customerID,
    gender,
    CAST(SeniorCitizen AS INTEGER)                      AS SeniorCitizen,
    Partner,
    Dependents,
    CAST(tenure AS INTEGER)                             AS tenure,
    PhoneService,
    -- "No phone service" / "No internet service" are already encoded by
    -- PhoneService / InternetService, so collapse them to "No"
    CASE WHEN MultipleLines    = 'No phone service'    THEN 'No' ELSE MultipleLines    END AS MultipleLines,
    InternetService,
    CASE WHEN OnlineSecurity   = 'No internet service' THEN 'No' ELSE OnlineSecurity   END AS OnlineSecurity,
    CASE WHEN OnlineBackup     = 'No internet service' THEN 'No' ELSE OnlineBackup     END AS OnlineBackup,
    CASE WHEN DeviceProtection = 'No internet service' THEN 'No' ELSE DeviceProtection END AS DeviceProtection,
    CASE WHEN TechSupport      = 'No internet service' THEN 'No' ELSE TechSupport      END AS TechSupport,
    CASE WHEN StreamingTV      = 'No internet service' THEN 'No' ELSE StreamingTV      END AS StreamingTV,
    CASE WHEN StreamingMovies  = 'No internet service' THEN 'No' ELSE StreamingMovies  END AS StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    CAST(MonthlyCharges AS REAL)                        AS MonthlyCharges,
    -- TotalCharges is text in the raw file; 11 brand-new customers (tenure 0) have blanks
    COALESCE(CAST(NULLIF(TRIM(TotalCharges), '') AS REAL), 0.0) AS TotalCharges,
    CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END           AS Churn
FROM raw_customers
WHERE customerID IS NOT NULL;
