
# Movie Rating Prediction Using Python

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "IMDb Movies India.csv"
OUTPUT_DIR = BASE_DIR

# 1. Load the dataset
df = pd.read_csv(DATA_PATH, encoding="latin1")

print("Original Dataset:")
print(df.head())

# 2. Data Cleaning
# Remove duplicate records
df = df.drop_duplicates()

# Convert important columns into numbers
for col in ["Rating", "Year", "Duration", "Votes"]:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"(\d+\.?\d*)", expand=False),
            errors="coerce"
        )

# Remove movies without ratings
df = df.dropna(subset=["Rating"])

# Save cleaned data
df.to_csv(OUTPUT_DIR / "cleaned_movies.csv", index=False)

print("\nData Cleaning Completed!")
print("Cleaned Dataset Shape:", df.shape)
print(df.head())

# 3. Select features for prediction
features = ["Year", "Duration", "Votes"]

# Check that required columns exist
for col in features:
    if col not in df.columns:
        raise ValueError(f"Required column is missing: {col}")

# Fill missing feature values with their median
for col in features:
    df[col] = df[col].fillna(df[col].median())

# Prepare input and output
X = df[features]
y = df["Rating"]

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train the Machine Learning model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# 6. Predict movie ratings
predictions = model.predict(X_test)

# 7. Check model performance
error = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", round(error, 2))

# 8. Draw a graph
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel("Actual Movie Rating")
plt.ylabel("Predicted Movie Rating")
plt.title("Actual vs Predicted Movie Ratings")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rating_predictions.png")
plt.show()

# 9. Predict a rating for a new movie
new_movie = pd.DataFrame({
    "Year": [2025],
    "Duration": [120],
    "Votes": [5000]
})

predicted_rating = model.predict(new_movie)

print("\nPredicted Rating for New Movie:")
print(round(predicted_rating[0], 2))
