# Credit Card Fraud Detection

This project builds a machine learning pipeline to detect fraudulent credit card transactions from `creditcard.csv`.

## What it does

- Loads the transaction dataset and separates features from the `Class` target.
- Splits the data into stratified training and testing sets.
- Preprocesses numeric features with median imputation and standard scaling.
- Trains multiple classifiers for comparison.
- Handles class imbalance with class weights and undersampling.
- Evaluates results using precision, recall, F1-score, confusion matrix, and a full classification report.

## Models used

- Logistic Regression with `class_weight='balanced'`
- Random Forest with `class_weight='balanced_subsample'`
- Logistic Regression with `RandomUnderSampler`

## Requirements

Install the Python packages used by the script:

```bash
pip install pandas scikit-learn imbalanced-learn
```

## Project files

- `creditcard.csv` - transaction dataset
- `fraud_detection_model.py` - training and evaluation script

## How to run

From the project folder:

```bash
python fraud_detection_model.py
```

If you are using the workspace virtual environment on Windows, you can also run:

```bash
.venv\Scripts\python.exe fraud_detection_model.py
```

## Expected output

The script prints:

- class distribution for train and test splits
- precision, recall, and F1-score for each model
- confusion matrix
- classification report

## Notes

- The dataset is highly imbalanced, so accuracy alone is not a useful metric.
- Random forest with class weighting usually performs strongly on this dataset.
- Undersampling can improve recall for the fraud class, but may reduce precision.

## Dataset column

The target label is `Class`:

- `0` = genuine transaction
- `1` = fraudulent transaction
