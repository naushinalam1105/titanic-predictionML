import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# load dataset
df = pd.read_csv("housing.csv")

# find the price target column regardless of case
target_col = None
for col in ['price', 'SalePrice', 'Price', 'Median_House_Value']:
    if col in df.columns:
        target_col = col
        break

if not target_col:
    # fallback to the first numeric column with 'price' in its name
    price_candidates = [c for c in df.columns if 'price' in c.lower()]
    target_col = price_candidates[0] if price_candidates else df.select_dtypes(include=[np.number]).columns[-1]

print(f"Target column identified: {target_col}")

# convert categorical text columns (like yes/no, furnishingstatus) to numeric dummies
df_encoded = pd.get_dummies(df, drop_first=True)

# handle missing values
df_encoded = df_encoded.fillna(df_encoded.median())

# separate features and target
X = df_encoded.drop(columns=[target_col])
if 'Id' in X.columns:
    X = X.drop(columns=['Id'])

y = df_encoded[target_col]

# train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# predict on test set
y_pred = model.predict(X_test)

# evaluate model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# plot predicted vs actual values
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='darkorange', edgecolor='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2)
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()
print("Plot saved to actual_vs_predicted.png")