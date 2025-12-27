🛡️ Bot Detection & Anomaly Analysis for E-commerce Traffic
📌 Project Overview

Bots are increasingly capable of mimicking human behavior on websites, making simple rule-based detection unreliable over time.
This project builds a hybrid bot detection system that combines:

Explainable rule-based logic (precision-first)

Unsupervised machine learning (recall-focused)

Behavioral analytics derived from real user session data

The goal is not perfect accuracy, but a production-ready detection pipeline that balances risk, trust, and user experience.

🎯 Problem Statement

E-commerce platforms must detect automated traffic (bots) without accidentally blocking genuine users.

Key challenges:

Bots can fake individual human signals

Rule-based systems miss adaptive bots

Pure ML systems introduce false positives

This project addresses those challenges using a layered detection strategy.

🧠 Detection Strategy
1️⃣ Rule-Based Detection (Baseline)

Rules capture obvious bot behavior using:

Request velocity

Scroll activity

Mouse movement patterns

✅ Very high precision
❌ Misses subtle, adaptive bots

2️⃣ Unsupervised Machine Learning

To detect unknown or evolving bot behavior, the following models were used:

Isolation Forest (anomaly detection)

One-Class SVM (human behavior boundary learning)

Models are trained on human-only baselines, allowing them to flag sessions that quietly deviate from normal behavior.

3️⃣ Hybrid Decision Layer (Final System)

The final classification combines:

Rule-based confidence

ML anomaly signals

Anomaly score thresholds

This mirrors how real-world fraud and bot-detection systems are designed.

📊 Exploratory Data Analysis (EDA)

EDA focused on understanding behavioral differences between humans and bots and validating feature usefulness.

Key insights:

Bots show higher request velocity

Humans exhibit richer scroll and mouse interaction

Significant overlap exists → justifies ML usage

Below are the final 4 EDA outputs used to guide modeling decisions.

🔹 1. Requests per Second — Human vs Bot

Shows clear separation in request velocity, with bots skewing higher.

🔹 2. Scroll Depth — Human vs Bot

Humans tend to scroll more naturally, while bots often show shallow or inconsistent scrolling.

🔹 3. Mouse Movement — Human vs Bot

Mouse activity is one of the strongest behavioral signals separating humans from bots.

🔹 4. Feature Correlation Matrix

Confirms low multicollinearity and validates feature independence for ML models.

🧪 Model Evaluation Summary
Model	Precision (Bot)	Recall (Bot)	Notes
Rule-Based	Very High	Moderate	Safe, conservative
Isolation Forest	Moderate	Very High	Captures subtle bots
One-Class SVM	Balanced	Balanced	Stable boundary model
Hybrid Model	Balanced	High	Production-ready

📌 The hybrid system improves recall without significantly harming user experience.

🛠️ Tech Stack

Python

Pandas / NumPy

Scikit-learn

Matplotlib

🚀 Key Learnings

Rules are excellent for known patterns

ML is necessary for adaptive behavior

Hybrid systems outperform single-method approaches

Fraud detection is about trade-offs, not perfection

🔮 Future Enhancements

Risk scoring instead of binary labels

Cost-sensitive evaluation

Temporal behavior modeling

Drift detection as bots evolve

👤 Author

Anuj Upadhyay
Data Analyst |
🔗 LinkedIn: (https://www.linkedin.com/in/anuj-upadhyay-1b040b29/)

Behavioral Feature Engineering

Unsupervised ML
