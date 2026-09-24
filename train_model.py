import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# 1. DATA COLLECTION
# -----------------------------

df = pd.read_csv("data/house_data.csv")

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())


# -----------------------------
# 2. DATA PREPROCESSING
# -----------------------------

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows containing missing values
df = df.dropna()

print("\nDataset shape after preprocessing:", df.shape)


# -----------------------------
# 3. FEATURE SELECTION
# -----------------------------

features = [
    "area",
    "bedrooms",
    "bathrooms",
    "floors",
    "parking",
    "age"
]

X = df[features]
y = df["price"]


# -----------------------------
# 4. TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 5. MODEL BUILDING
# -----------------------------

model = LinearRegression()

model.fit(X_train, y_train)


# -----------------------------
# 6. PREDICTION
# -----------------------------

y_pred = model.predict(X_test)


# -----------------------------
# 7. MODEL EVALUATION
# -----------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"MAE  : {mae:,.2f}")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R2   : {r2:.4f}")


# -----------------------------
# 8. SAVE MODEL
# -----------------------------

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/house_price_model.pkl")

print("\nModel saved successfully!")
print("Location: model/house_price_model.pkl")