# IEEE-CIS Fraud Detection - Final Report

## Objective
Predict potentially fraudulent online transactions.

## Dataset
Training transactions: 590,540
Test transactions: 506,691
Fraud rate: approximately 3.50%

## Data Integration
- train_transaction.csv
- train_identity.csv
- test_transaction.csv
- test_identity.csv

Joined using TransactionID.

## Data Preparation
- Missing value analysis
- Class imbalance investigation
- Selected transaction and identity variables
- Categorical encoding
- Numeric imputation
- Chronological validation split

## Initial Models

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.833014 | 0.191693 |
| Random Forest | 0.875946 | 0.457162 |
| HistGradientBoosting | 0.901169 | 0.511497 |

## Feature Engineering
Original features: 144
Final features: 160

Added:
- Missing-value counts
- Card frequency statistics
- Address frequency statistics
- Email and device frequencies
- Card-address combination frequencies
- Transaction amount group ratios
- Transaction amount z-scores

## Final Model Comparison

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Original HGB | 0.901169 | 0.511497 |
| HGB + Fraud FE | 0.902538 | 0.515967 |
| HGB + Fraud FE V2 | 0.912001 | 0.534559 |

## Selected Model
HistGradientBoosting + Fraud Feature Engineering V2

## Validation Results
ROC-AUC: 0.912001
PR-AUC: 0.534559

ROC-AUC improvement: 0.010832
PR-AUC improvement: 0.023062

## Validation Methodology
Training sample: 250,000
Validation transactions: 118,108

Transactions were split chronologically.
Validation feature statistics came from
the earlier training period.

## Limitations
- Selected subset of original features
- High class imbalance
- Many missing identity attributes
- Single chronological validation split
- Validation reused for model selection
- No independently labeled final test evaluation
- No official Kaggle leaderboard score
- Model scores are not necessarily calibrated
- Not for real-world automated fraud decisions

## Competition
https://www.kaggle.com/competitions/ieee-fraud-detection
