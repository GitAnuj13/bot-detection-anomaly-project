# ============================================================
# Bot & Fraud Detection using Rule-Based + ML (Isolation Forest, OCSVM)
# ============================================================

import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 1. Load Data
# ============================================================

DATA_PATH = "../processed/session_features_intermediate.csv"
session_features = pd.read_csv(DATA_PATH)

# Ground truth from user_id convention
session_features["actual_label"] = session_features["user_id"].str[0].map(
    {"U": "Human", "B": "Bot", "M": "Bot", "A": "Bot"}
)

# ============================================================
# 2. Base Feature Set (used by rules + ML baseline)
# ============================================================

base_features = [
    "requests_per_sec",
    "avg_scroll_depth",
    "avg_mouse_movements"
]

X_base = (
    session_features[base_features]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

# ============================================================
# 3. Baseline Isolation Forest (Unsupervised)
# ============================================================

baseline_if = IsolationForest(
    n_estimators=100,
    contamination=0.29,
    random_state=42
)

baseline_if.fit(X_base)

session_features["if_baseline_pred"] = baseline_if.predict(X_base)
session_features["if_baseline_bot"] = (
    session_features["if_baseline_pred"] == -1
).astype(int)

# ============================================================
# 4. Feature Engineering for Advanced ML
# ============================================================

session_features["pages_per_min"] = (
    session_features["total_pages"] /
    (session_features["session_duration_sec"] / 60).replace(0, np.nan)
).fillna(0)

session_features["scroll_per_sec"] = (
    session_features["avg_scroll_depth"] /
    session_features["session_duration_sec"].replace(0, np.nan)
).fillna(0)

ml_features = [
    "requests_per_sec",
    "pages_per_min",
    "avg_scroll_depth",
    "avg_mouse_movements",
    "scroll_per_sec"
]

X_all = (
    session_features[ml_features]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

# ============================================================
# 5. Train ML Models on Human Baseline
# ============================================================

human_mask = session_features["classification"].eq("Human")

X_train = X_all.loc[human_mask]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_all_scaled = scaler.transform(X_all)

# ---------------------------
# Improved Isolation Forest
# ---------------------------

improved_if = IsolationForest(
    n_estimators=200,
    contamination=0.15,
    random_state=42,
    n_jobs=-1
)

improved_if.fit(X_train_scaled)

session_features["if_pred"] = improved_if.predict(X_all_scaled)
session_features["if_anomaly_score"] = improved_if.score_samples(X_all_scaled)
session_features["if_bot"] = (session_features["if_pred"] == -1).astype(int)

# ---------------------------
# One-Class SVM
# ---------------------------

ocsvm = OneClassSVM(
    kernel="rbf",
    nu=0.10,
    gamma="scale"
)

ocsvm.fit(X_train_scaled)

session_features["svm_pred"] = ocsvm.predict(X_all_scaled)
session_features["svm_bot"] = (session_features["svm_pred"] == -1).astype(int)

# ============================================================
# 6. Final Hybrid Decision Logic
# ============================================================

rule_bot = session_features["classification"].isin(
    ["Bot", "Suspicious"]
).astype(int)

ml_bot = (
    (session_features["if_bot"] == 1) |
    (session_features["svm_bot"] == 1)
).astype(int)

session_features["final_bot"] = (
    (rule_bot == 1) |
    (
        (ml_bot == 1) &
        (session_features["if_anomaly_score"] < -0.55)
    )
).astype(int)

# ============================================================
# 7. Model Evaluation
# ============================================================

y_true = (session_features["actual_label"] == "Bot").astype(int)

print("\n================ FINAL ENSEMBLE CLASSIFICATION REPORT ================\n")

print("--- RULE BASED ---")
print(classification_report(y_true, rule_bot))

print("\n--- ISOLATION FOREST ---")
print(classification_report(y_true, session_features["if_bot"]))

print("\n--- ONE CLASS SVM ---")
print(classification_report(y_true, session_features["svm_bot"]))

print("\n--- HYBRID MODEL ---")
print(classification_report(y_true, session_features["final_bot"]))
