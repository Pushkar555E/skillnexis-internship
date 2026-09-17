# SkillNexis Machine Learning and AI Internship

This repository contains my assignments and project work from the SkillNexis Machine Learning and AI internship.

## Internship Progress

| Week | Topic | Status |
| --- | --- | --- |
| Week 1 | ML fundamentals and Titanic data preprocessing | Completed |
| Week 2 | House price prediction with linear regression | Completed |
| Week 3 | Coming soon | Pending |
| Week 4 | Coming soon | Pending |

## Repository Structure

```text
skillnexis-internship/
|-- Week-1/
|   |-- assignment.ipynb
|   |-- Titanic-Dataset.csv
|   |-- Titanic-Cleaned-Dataset.csv
|   |-- download.png
|   `-- README.md
|-- Week-2/
|   |-- complete_house_price_model.py
|   |-- House_Price_Prediction_Completed_verified.ipynb
|   |-- Housing.csv
|   |-- house_price_predictions.csv
|   `-- actual_vs_predicted.png
`-- README.md
```

## Technologies

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Week 1: Titanic Data Cleaning

The Week 1 project covers dataset exploration, missing-value handling, categorical encoding, visualization, and export of a cleaned Titanic dataset.

[View the Week 1 project](./Week-1/)

## Week 2: House Price Prediction

The Week 2 project trains a linear regression model to predict house prices from `Housing.csv`.

The workflow includes:

- Separating features and the target price
- Detecting categorical and numerical features
- Encoding categorical features with `OneHotEncoder`
- Building a preprocessing and regression pipeline
- Evaluating the model with MSE and R2 score
- Saving predictions to `house_price_predictions.csv`
- Creating an actual-versus-predicted plot

Run the Python script from the `Week-2` directory:

```bash
pip install pandas matplotlib scikit-learn
python complete_house_price_model.py
```

[View the Week 2 project](./Week-2/)
