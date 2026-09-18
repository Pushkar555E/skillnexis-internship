# Bank Loan Approval Prediction — Option 2

This submission implements the Week 4 Option 2 project: **Bank Loan Approval Prediction**.

## Source and workflow
The project uses the supplied Loan Prediction dataset and follows the Week 4 workflow:
- Clean categorical data (Gender, Marital Status, Income)
- Train a Random Forest model
- Evaluate with a Confusion Matrix
- Predict whether a loan can be approved

## Dataset
- Rows: 614
- Columns: 13
- Target: `Loan_Status`
- `Y` = approved, `N` = rejected

## Model
Random Forest Classifier, 300 trees, random_state=42, with balanced class weights.

## Preprocessing
- Numerical missing values: median imputation
- Categorical missing values: most-frequent imputation
- Categorical variables: one-hot encoding
- `Loan_ID` is excluded because it is an identifier rather than a predictive feature.

## Evaluation
Accuracy: 0.8130
Precision: 0.8370
Recall: 0.9059

Confusion matrix:
[[23 15]
 [ 8 77]]

## Files
- `loan_prediction_dataset.csv` — dataset
- `bank_loan_approval.py` — reproducible ML script
- `bank_loan_approval_report.docx` — documentation/report
- `bank_loan_approval_presentation.pptx` — presentation
- `confusion_matrix.png` — evaluation visualization
- `loan_status_distribution.png` — target distribution
- `feature_importance.png` — model feature importance
- `sample_predictions.csv` — example predictions
- `results.txt` — numerical results
- `requirements.txt` — Python dependencies

## Run
1. Install dependencies: `pip install -r requirements.txt`
2. Put the CSV in the same folder as the Python script.
3. Run: `python bank_loan_approval.py`
