# -------------------------------------------------------------
# Program: Univariate, Bivariate, and Multivariate Regression
# Dataset: Iris Dataset
# -----------------Expt 1--------------------------------------------
# Aim:
# To implement a Python program for performing univariate,
# bivariate, and multivariate regression using the Iris dataset.
# -------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
# Step 1: Import necessary libraries
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 2: Read the dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

print("\nFirst five rows of the Iris dataset:\n")
print(data.head())

# -------------------------------------------------------------
# Step 3: Prepare the data
# -------------------------------------------------------------
# Let's predict 'sepal length (cm)' based on other features.
y = data['sepal length (cm)']

# -------------------------------------------------------------
# Step 4: Univariate Regression
# -------------------------------------------------------------
print("\n=== UNIVARIATE REGRESSION ===")

X_uni = data[['petal length (cm)']]  # one independent variable

model_uni = LinearRegression()
model_uni.fit(X_uni, y)

y_pred_uni = model_uni.predict(X_uni)
r2_uni = r2_score(y, y_pred_uni)

print(f"Coefficient (slope): {model_uni.coef_[0]:.4f}")
print(f"Intercept: {model_uni.intercept_:.4f}")
print(f"R² value: {r2_uni:.4f}")

# Plotting univariate regression
plt.figure(figsize=(6, 4))
plt.scatter(X_uni, y, color='blue', label='Actual Data')
plt.plot(X_uni, y_pred_uni, color='red', label='Regression Line')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Sepal Length (cm)')
plt.title('Univariate Linear Regression')
plt.legend()
plt.show()

# -------------------------------------------------------------
# Step 5: Bivariate Regression
# -------------------------------------------------------------
print("\n=== BIVARIATE REGRESSION ===")

X_bi = data[['petal length (cm)', 'petal width (cm)']]

model_bi = LinearRegression()
model_bi.fit(X_bi, y)

y_pred_bi = model_bi.predict(X_bi)
r2_bi = r2_score(y, y_pred_bi)

print(f"Coefficients (slopes): {model_bi.coef_}")
print(f"Intercept: {model_bi.intercept_:.4f}")
print(f"R² value: {r2_bi:.4f}")

# 3D Plot for Bivariate Regression
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_bi['petal length (cm)'], X_bi['petal width (cm)'], y, color='blue', label='Actual')

# Plane equation: z = a*x + b*y + c
x_surf, y_surf = np.meshgrid(np.linspace(X_bi['petal length (cm)'].min(), X_bi['petal length (cm)'].max(), 20),
                             np.linspace(X_bi['petal width (cm)'].min(), X_bi['petal width (cm)'].max(), 20))
z_surf = (model_bi.coef_[0] * x_surf) + (model_bi.coef_[1] * y_surf) + model_bi.intercept_
ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.4, color='red')

ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')
ax.set_zlabel('Sepal Length (cm)')
plt.title('Bivariate Linear Regression (3D)')
plt.legend()
plt.show()

# -------------------------------------------------------------
# Step 6: Multivariate Regression
# -------------------------------------------------------------
print("\n=== MULTIVARIATE REGRESSION ===")

X_multi = data[['sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]

model_multi = LinearRegression()
model_multi.fit(X_multi, y)

y_pred_multi = model_multi.predict(X_multi)
r2_multi = r2_score(y, y_pred_multi)

print(f"Coefficients (slopes): {model_multi.coef_}")
print(f"Intercept: {model_multi.intercept_:.4f}")
print(f"R² value: {r2_multi:.4f}")

# Plot Predicted vs Actual
plt.figure(figsize=(6, 4))
plt.scatter(y, y_pred_multi, color='green')
plt.xlabel('Actual Sepal Length (cm)')
plt.ylabel('Predicted Sepal Length (cm)')
plt.title('Multivariate Regression: Actual vs Predicted')
plt.grid(True)
plt.show()

# -------------------------------------------------------------
# Step 7: Display Summary
# -------------------------------------------------------------
print("\n========== SUMMARY ==========")
print(f"Univariate R²:   {r2_uni:.4f}")
print(f"Bivariate R²:    {r2_bi:.4f}")
print(f"Multivariate R²: {r2_multi:.4f}")
print("=================================")