# Week 1: Titanic Survival Prediction and Data Cleaning

This project is part of my **Machine Learning and AI Internship**.

The goal for Week 1 was to review the fundamentals of machine learning and practice basic data preprocessing with the Titanic dataset.

## Topics Covered

* Machine Learning fundamentals
* Importing and exploring datasets
* Data cleaning
* Handling missing data
* Feature preprocessing
* Encoding categorical variables
* Data visualization
* Preparing a dataset for Machine Learning

## Dataset

The project uses the **Titanic Dataset**.

The dataset contains information about Titanic passengers, including:

* Passenger ID
* Survival status
* Passenger class
* Name
* Sex
* Age
* Number of siblings/spouses aboard
* Number of parents/children aboard
* Ticket
* Fare
* Cabin
* Port of embarkation

## Tasks Completed

### 1. Dataset Exploration

The dataset was loaded using Pandas and explored using:

* `df.info()`
* `df.describe()`
* `df.isnull().sum()`

This was used to understand the structure of the dataset and identify missing values.

### 2. Handling Missing Data

Missing values were handled as follows:

* Missing `Age` values were replaced using the **median age**.
* Missing `Embarked` values were replaced using the **mode**.
* The `Cabin` column was removed because a large proportion of its values were missing.

### 3. Encoding Categorical Variables

The categorical columns were converted into numerical representations.

* `Sex` was encoded using `LabelEncoder`.
* `Embarked` was encoded using one-hot encoding.

### 4. Data Visualization

A histogram was created using Matplotlib to visualize the distribution of passenger ages.

### 5. Cleaned Dataset

After preprocessing, the cleaned dataset was exported as:

`Titanic-Cleaned-Dataset.csv`

## Project Files

| File                           | Description                         |
| ------------------------------ | ----------------------------------- |
| `Titanic-Dataset.csv`          | Original Titanic dataset            |
| `Titanic-Cleaned-Dataset.csv`  | Dataset after preprocessing         |
| `assignment.ipynb`             | Notebook containing the assignment |
| `download.png`                 | Age distribution visualization     |
| `README.md`                    | Project documentation               |

## Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn

## How to Run

Install the required Python libraries:

```bash
pip install pandas matplotlib scikit-learn
```

Then open `assignment.ipynb` in Jupyter Notebook or VS Code and run the cells.

Make sure `Titanic-Dataset.csv` is in the same directory as the notebook.

## Result

The Titanic dataset was cleaned and prepared for future machine learning tasks.

I handled missing values, encoded categorical variables, visualized the age distribution, and exported the cleaned data as a new CSV file.

## Internship

**Machine Learning and AI Internship - Week 1**

Focus:

**ML Fundamentals + Data Preprocessing**
