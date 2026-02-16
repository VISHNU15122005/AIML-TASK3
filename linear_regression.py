# ==========================================
# TASK 3 - LINEAR REGRESSION (FINAL FIXED)
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# STEP 1 - Load Dataset
# -------------------------------
df = pd.read_csv("house_data.csv")

print("\nColumns:", df.columns)
print(df.head())

# -------------------------------
# STEP 2 - Clean Data
# -------------------------------
df = df.dropna()
df = df.drop_duplicates()

# -------------------------------
# STEP 3 - Convert TEXT → NUMBERS
# -------------------------------
df = pd.get_dummies(df, drop_first=True)

print("\nAfter Encoding Columns:", df.columns)

# -------------------------------
# STEP 4 - Select Features & Target
# Last column = target
# -------------------------------
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("\nFeatures:", X.columns)
print("Target:", y.name)

# -------------------------------
# STEP 5 - Split Data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 6 - Train Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Trained Successfully")

# -------------------------------
# STEP 7 - Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# STEP 8 - Evaluation
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Results")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# -------------------------------
# STEP 9 - Plot
# -------------------------------
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()
