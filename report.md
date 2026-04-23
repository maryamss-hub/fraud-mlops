## Fraud Detection MLOps Research Report

### Title Page

- **Course**: MLOps BS DS  
- **Author**: Maryam Khalid  
- **Date**: April 2026  

---

## Abstract

This report presents an end-to-end MLOps implementation for fraud detection using the IEEE-CIS style transaction dataset. The system covers pipeline orchestration and experiment tracking, robust data handling, model development and comparison, cost-sensitive optimization, CI/CD automation, monitoring and alerting, drift simulation, retraining strategy design, and model explainability. Due to resource constraints, the originally planned Kubeflow workflow was replaced with an MLflow-based implementation (approved by the instructor). Minikube infrastructure components were still deployed to demonstrate Kubernetes concepts, including a dedicated namespace, persistent volume claim (PVC), and resource quotas. Across the conducted experiments, LightGBM delivered the best overall performance among tested models, while cost-sensitive learning substantially reduced estimated business losses. Monitoring and drift detection confirmed the feasibility of continuous evaluation and retraining triggers in production-like settings.

---

## System Architecture Overview

The system architecture follows a standard MLOps lifecycle:

- **Data layer**: Raw transactions and identity tables are ingested and merged; preprocessing addresses missingness and categorical encoding; feature engineering produces model-ready tabular features.
- **Experiment layer**: MLflow tracks experiments and artifacts across seven nested pipeline stages.
- **Modeling layer**: Candidate models (XGBoost, LightGBM, Hybrid RF+XGBoost) are trained and evaluated using classification metrics and AUC-ROC.
- **Optimization layer**: Cost-sensitive training is applied to reduce false-negative business impact.
- **Delivery layer**: CI/CD automates linting, testing, container builds, and (where applicable) deployment steps.
- **Observability layer**: FastAPI serves inference; Prometheus collects metrics and evaluates alert rules; Grafana dashboards provide operational visibility.
- **Governance layer**: Drift detection and retraining strategies define when to refresh the model; explainability supports transparency and stakeholder trust.

**Dataset summary**: 590k transactions, 3.49% fraud rate, 219 features.

---

## Task 1 — MLflow Pipeline Setup

**Important note (Task 1)**: Kubeflow was replaced with MLflow (approved by instructor due to resource constraints). Minikube was still deployed with namespace, PVC and quotas.

**Infrastructure and cluster constraints**:

- Minikube deployed
- Namespace: `fraud-detection`
- PVC: `fraud-artifacts-pvc` (10Gi, bound)
- Resource quotas: 4CPU / 8Gi

**MLflow pipeline implementation**:

- MLflow used for orchestration with experiment: `fraud-detection`
- Seven nested MLflow runs (stages):
  - Data Ingestion
  - Data Validation
  - Preprocessing
  - Feature Engineering
  - Model Training
  - Model Evaluation
  - Conditional Deployment
- **Model registration policy**: model only registered if **recall > 0.85**
- **Artifacts logged**: confusion matrix, ROC curve, SHAP plot

---

## Task 2 — Data Challenges

The dataset presents typical fraud-detection constraints: class imbalance (3.49% fraud), heterogeneous feature types, missingness, and high-cardinality categoricals. The adopted data handling strategy was intentionally simple, explainable, and operationally robust:

- **Missing values**:
  - Numeric: median imputation
  - Categorical: imputed as `"Unknown"`
- **Categorical encoding**:
  - Target encoding for **card1–card6** and **email domains**

Two imbalance-handling strategies were evaluated with XGBoost:

- **SMOTE+XGBoost**: Precision 0.197, Recall 0.823, F1 0.318, AUC 0.926  
- **ClassWeight+XGBoost**: Precision 0.244, Recall 0.819, F1 0.376, AUC 0.937  

**Result**: ClassWeight performed better overall.

---

## Task 3 — Model Comparison

Three model families were compared under a consistent evaluation protocol. The results show that high precision does not necessarily imply strong fraud capture; recall is critical due to the cost of missed fraud.

### Results (metrics)

| Model | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|
| LightGBM | 0.927 | 0.530 | 0.675 | 0.957 |
| XGBoost | 0.905 | 0.426 | 0.580 | 0.931 |
| Hybrid RF+XGBoost | 0.897 | 0.413 | 0.566 | 0.926 |

**Result**: LightGBM best overall.

---

## Task 4 — Cost-Sensitive Learning

Cost-sensitive learning targets asymmetric error costs, especially the large financial impact of false negatives in fraud detection. Two training configurations were compared.

### Business impact results

| Configuration | Fraud loss | False alarms | Total |
|---|---:|---:|---:|
| Standard XGBoost | $1,743,500 | $2,710 | $1,746,210 |
| Cost-sensitive XGBoost (scale_pos_weight=50) | $389,000 | $256,440 | $645,440 |

**Result**: 63% cost reduction.

---

## Task 5 — CI/CD Pipeline

The CI/CD system was implemented as a GitHub Actions pipeline with four stages and triggers on push and pull request:

- **4-stage pipeline**
- **lint-and-test**: PASSED  
- **build-docker**: PASSED (training + inference images built and pushed to GHCR)  
- **deploy-kubeflow**: failed due to missing `KFP_HOST` secret (no live cluster — expected)  
- **intelligent-retrain**: skipped  

This design supports reproducibility (tests and linting), delivery automation (container builds), and controlled deployment behavior when external secrets or a cluster are unavailable.

---

## Task 6 — Monitoring System

Operational monitoring was implemented using FastAPI, Prometheus, and Grafana:

- FastAPI inference API running at **port 8000**
- Prometheus metrics endpoint at **`/metrics`** exposing:
  - `fraud_api_requests_total`
  - `fraud_api_latency_seconds`
  - `fraud_model_recall`
  - `feature_drift_score`

**Prometheus alert rules** defined for:

- recall < 0.80
- drift > 0.2
- latency > 0.5s

**Grafana dashboards**:

1. **System Health**: request rate, latency, error rate, CPU  
2. **Model Performance**: recall over time, precision, fraud detection rate, confidence distribution  
3. **Data Drift**: PSI scores, missing value rate, drift alert history  

---

## Task 7 — Drift Simulation

Drift was simulated using a **time-based split on `TransactionDT`**. PSI analysis identified strongly drifted features consistent with temporal partitioning and identity/card dynamics.

### Top drifted features (PSI)

| Feature | PSI |
|---|---:|
| TransactionID | 12.43 |
| TransactionDT | 12.43 |
| id_33 | 12.42 |
| card4 | 10.14 |
| id_30 | 5.39 |

**Result**: `should_retrain()` returned **True**.

---

## Task 8 — Retraining Strategy

Three retraining strategies were compared to balance operational cost and model stability.

### Strategy comparison

| Strategy | Cost | Stability | Trigger logic |
|---|---|---|---|
| Threshold-based | medium cost | medium stability | triggers on metric drop |
| Periodic | high cost | high stability | fixed weekly schedule |
| Hybrid (recommended) | medium cost | high stability | combines emergency trigger with weekly schedule |

**Recommendation**: Hybrid (recommended).

---

## Task 9 — Explainability

Model explainability was performed using **TreeExplainer on XGBoost**. Global SHAP analysis identified the features most influential to fraud prediction:

- Top features: **C13 (most important)**, C1, C14, TransactionAmt, V70, P_emaildomain

**Interpretation**: C13 represents behavioral aggregation signals most predictive of fraud, indicating that aggregated transaction behavior provides substantial discriminatory power beyond single raw fields. Such insights support both stakeholder communication and targeted feature monitoring for drift.

---

## Conclusion

This project demonstrates a complete MLOps workflow for fraud detection: pipeline tracking and orchestration, robust data preprocessing and encoding, comparative modeling, business-cost optimization, CI/CD automation, production-grade monitoring, drift-driven retraining triggers, and explainability. The MLflow implementation successfully replaces Kubeflow/KFP under realistic constraints while preserving rigorous experiment tracking and artifact management. Empirically, LightGBM achieved the best overall predictive performance among evaluated models, and cost-sensitive XGBoost produced a substantial reduction in estimated operational losses. The monitoring and drift simulation results further reinforce the need for continuous evaluation and a hybrid retraining strategy to sustain performance in dynamic fraud environments.

