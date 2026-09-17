import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. Load dataset
DATA_PATH = "Housing.csv"
df = pd.read_csv(DATA_PATH)

# 2. Separate features and target
X = df.drop(columns=["price"])
y = df["price"]

# 3. Identify categorical and numerical columns
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

# 4. Preprocess categorical variables with one-hot encoding.
#    Numeric variables are passed through unchanged.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", drop="first"),
            categorical_features,
        ),
        ("numerical", "passthrough", numerical_features),
    ]
)

# 5. Build Linear Regression pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 7. Train the model
model.fit(X_train, y_train)

# 8. Predict house prices on the test set
y_pred = model.predict(X_test)

# 9. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Dataset shape:", df.shape)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"R² Score: {r2:.4f}")

# 10. Display actual vs predicted values
results = pd.DataFrame(
    {
        "Actual Price": y_test.to_numpy(),
        "Predicted Price": y_pred,
    }
).reset_index(drop=True)

print("\nFirst 10 actual vs predicted values:")
print(results.head(10).to_string(index=False))

# 11. Save predictions
results.to_csv("house_price_predictions.csv", index=False)

# 12. Plot predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())
plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()
