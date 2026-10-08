Credit Card Fraud Detection: SQL, EDA, Modeling and Business Impact
Detects fraudulent card transactions in a heavily imbalanced dataset (0.17% fraud) and translates model performance into business terms: how much fraud value is stopped and how many transactions analysts must review.
Tools: Python (pandas, scikit-learn, XGBoost, imbalanced-learn, SHAP, seaborn), SQL (SQLite)
Business problem
Missing fraud costs money; flagging too many normal transactions wastes analyst time and annoys customers. The goal is a model and an alert threshold that catch most fraud with very few false alarms.
Data
Kaggle: Credit Card Fraud Detection (284,807 transactions over two days, European cardholders). `V1`-`V28` are anonymized PCA components; `Time` is seconds since the first transaction. After removing 1,081 exact duplicates: 283,726 rows, 473 fraud (0.167%).
Approach
Cleaning: removed duplicates (they can leak across train/test).
SQL analysis: overall fraud summary, largest fraud cases, fraud rate by amount band and by hour using CTEs and window functions.
EDA: class balance, amount and time patterns, signed correlations, correlation heatmap, feature distributions by class.
Modeling: stratified train/test split; scaling and resampling inside a pipeline (no leakage); 9 models (Logistic Regression, Random Forest, XGBoost, Neural Network; with class weights, SMOTE or undersampling) compared with 5-fold CV on the training set using AUC-PR, with ROC/PR curves and confusion matrices for every model.
Threshold tuning on out-of-fold training predictions (test set untouched).
Single test-set evaluation with a bootstrap confidence interval, plus a chronological-split robustness check.
Explainability: feature importance and SHAP summary for the final model.
Business impact with real transaction amounts and a review-cost sensitivity table.
Results
Metric	Value
Best model (CV)	XGBoost, cross-validated AUC-PR 0.85 ± 0.03
Test AUC-PR	0.83 (95% CI 0.77-0.90)
Precision / recall at tuned threshold	0.95 / 0.77
Frauds caught / missed / false alarms	73 / 22 / 4 (test set, 56,746 transactions)
Fraud value stopped	about 74%
Chronological split (future-data check)	AUC-PR 0.79, recall 0.74
Findings: fraud rate is highest for very small transactions (under 1 unit, 0.39%) and in low-traffic hours; XGBoost beats Random Forest, the neural network and Logistic Regression; `V14` and `V4` are the strongest signals (SHAP); resampling (SMOTE, undersampling) did not improve ranking quality and sharply reduced precision at the default threshold.
Limitations
Only 95 fraud cases in the test set (wide uncertainty), two days of data (no drift or seasonality), anonymized features (limited explainability), and illustrative business assumptions (review cost, full prevention of flagged fraud).
How to run
```bash
pip install -r requirements.txt
# put creditcard.csv next to the notebook, then:
jupyter notebook fraud\_detection\_project.ipynb
```
The full run takes about 20 minutes on one CPU core and much less on a multi-core machine or Colab.
Repository contents
```
fraud\_detection\_project.ipynb   main notebook (with outputs)
requirements.txt                Python dependencies
README.md                       this file
.gitignore                      keeps the 150 MB dataset out of the repo
```
