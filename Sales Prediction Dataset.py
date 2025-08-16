# Sales Prediction Analysis using Advertising Dataset (Fixed for Excel and Non-Numeric Columns)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# --- Step 1: Load Dataset (Supports Excel or CSV) ---
file_path = r'D:\Finlatics\MLResearch\Sales Prediction Dataset\advertising_sales_data.xlsx'

try:
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding='latin1')
    else:
        df = pd.read_excel(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file path.")
    exit()

print("Initial Data Preview:")
print(df.head())
print(df.info())

# --- Step 2: Clean Dataset ---
# Keep only numeric columns relevant to the analysis
numeric_cols = ['TV', 'Radio', 'Newspaper', 'Sales']
df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna()

print("\nCleaned Data Preview:")
print(df.head())
print(df.describe())

# --- Q1: Average TV Advertising ---
avg_tv = df['TV'].mean()
print(f"Average TV Advertising Spend: {avg_tv:.2f}")

# --- Q2: Correlation between Radio and Sales ---
radio_sales_corr = df['Radio'].corr(df['Sales'])
print(f"Correlation between Radio and Sales: {radio_sales_corr:.2f}")

# --- Q3: Highest Impact Advertising Medium ---
corr_with_sales = df.corr()['Sales'].drop('Sales')
print("\nCorrelation of each advertising medium with Sales:")
print(corr_with_sales)
max_corr_medium = corr_with_sales.idxmax()
print(f"Highest impact on sales: {max_corr_medium}")

sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# --- Step 3: Train-Test Split ---
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Linear Regression Model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Predictions and Q4 Visualization ---
y_pred = model.predict(X_test)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', edgecolors='k')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Model Performance:\nR2 Score: {r2:.2f}\nMSE: {mse:.2f}")

# --- Q5: Predict sales for new data ---
new_data = np.array([[200, 40, 50]])
new_prediction = model.predict(new_data)
print(f"Predicted Sales for TV=200, Radio=40, Newspaper=50: {new_prediction[0]:.2f}")

# --- Q6: Normalization Impact ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model_scaled = LinearRegression()
model_scaled.fit(X_train_s, y_train_s)
y_pred_s = model_scaled.predict(X_test_s)

r2_scaled = r2_score(y_test_s, y_pred_s)
print(f"R2 Score after Normalization: {r2_scaled:.2f}")

# --- Q7: Model with only Radio and Newspaper ---
X_rn = df[['Radio', 'Newspaper']]
X_train_rn, X_test_rn, y_train_rn, y_test_rn = train_test_split(X_rn, y, test_size=0.2, random_state=42)

model_rn = LinearRegression()
model_rn.fit(X_train_rn, y_train_rn)
y_pred_rn = model_rn.predict(X_test_rn)

r2_rn = r2_score(y_test_rn, y_pred_rn)
print(f"R2 Score using only Radio & Newspaper: {r2_rn:.2f}")
