from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DB_PATH = ROOT / "data" / "churn.db"
CLEAN_SQL = ROOT / "sql" / "clean.sql"
PROCESSED_CSV = ROOT / "data" / "processed" / "churn_clean.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
REPORTS = ROOT / "reports"

TARGET = "Churn"
ID_COL = "customerID"
SEED = 42

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "num_services"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
