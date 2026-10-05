"""Synthetic stand-in with the same schema as the IBM Telco CSV.
Used for CI / smoke tests only. For the real project, use the Kaggle file."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 4000
yn = lambda p: np.where(rng.random(n) < p, "Yes", "No")
tenure = rng.integers(0, 73, n)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .21, .24])
internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[.34, .44, .22])
monthly = np.round(np.where(internet == "No", rng.normal(25, 5, n), rng.normal(75, 20, n)).clip(18, 119), 2)
total = np.round(tenure * monthly, 2).astype(str)
total[tenure == 0] = " "  # blanks, like the real file
no_net = lambda p: np.where(internet == "No", "No internet service", yn(p))
tech, sec = no_net(.3), no_net(.3)
z = (-0.5 + 1.4 * (contract == "Month-to-month") - 0.04 * tenure + 0.8 * (internet == "Fiber optic")
     - 0.6 * (tech == "Yes") - 0.5 * (sec == "Yes") + 0.01 * (monthly - 65))
churn = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "Yes", "No")
df = pd.DataFrame({
    "customerID": [f"C{i:05d}" for i in range(n)],
    "gender": rng.choice(["Male", "Female"], n), "SeniorCitizen": rng.choice([0, 1], n, p=[.84, .16]),
    "Partner": yn(.48), "Dependents": yn(.3), "tenure": tenure, "PhoneService": yn(.9),
    "MultipleLines": rng.choice(["No", "Yes", "No phone service"], n),
    "InternetService": internet, "OnlineSecurity": sec, "OnlineBackup": no_net(.35),
    "DeviceProtection": no_net(.35), "TechSupport": tech,
    "StreamingTV": no_net(.4), "StreamingMovies": no_net(.4),
    "Contract": contract, "PaperlessBilling": yn(.6),
    "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n),
    "MonthlyCharges": monthly, "TotalCharges": total, "Churn": churn,
})
out = Path(__file__).resolve().parents[1] / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df.to_csv(out, index=False)
print("Wrote", out, f"churn={np.mean(churn == 'Yes'):.1%}")
