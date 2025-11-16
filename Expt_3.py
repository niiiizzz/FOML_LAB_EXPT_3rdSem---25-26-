# -------‐-------------------
#           expt 3
# ---------------------------
import numpy as np
import pandas as pd
from numpy import dot, exp, log, shape
from sklearn.linear_model import \
    LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Step 1: Load and Prepare Data ---
# FIX: The previous URL was broken (404 Error). Using a new, working link.
url = 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/Social_Network_Ads.csv'
try:
    # The dataset uses 'User ID', 'Gender', 'Age', 'EstimatedSalary', 'Purchased'
    # We will use 'Age' and 'EstimatedSalary' to predict 'Purchased'
    data = pd.read_csv(url)
    print("Successfully loaded SUV prediction dataset.")
    print("Dataset Head:")
    print(data.head())
    print("-" * 30)
except Exception as e:
    print(f"Failed to load data. Please check your internet connection. Error: {e}")
    exit()

# Select 'Age' (column 2) and 'EstimatedSalary' (column 3) as features (x)
# Select 'Purchased' (column 4) as the target (y)
x = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values

# Split data into training and testing sets ONCE for both models
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


# --- Part 1: Implementation using Scikit-Learn (In-built Function) ---
print("--- Part 1: Scikit-Learn Implementation ---")

# Feature Scaling
sc = StandardScaler()
x_train_scaled_sklearn = sc.fit_transform(x_train)
x_test_scaled_sklearn = sc.transform(x_test)

# Train the logistic regression model
classifier = SklearnLogisticRegression(random_state=0)
classifier.fit(x_train_scaled_sklearn, y_train)

# Make predictions
y_pred_sklearn = classifier.predict(x_test_scaled_sklearn)

# Evaluate the model
cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

print("Scikit-Learn Predictions (first 20):", y_pred_sklearn[:20])
print("Scikit-Learn Confusion Matrix:\n", cm_sklearn)
print(f"Scikit-Learn Accuracy: {acc_sklearn:.4f}")
print("-" * 50)


# --- Part 2: Implementation from Scratch (User-Defined Function) ---
print("\n--- Part 2: Custom Implementation from Scratch ---")

# Helper function to standardize features
def standardize(X):
    # Important: Create a copy to avoid modifying the original data
    X_std = np.copy(X).astype(float)
    for i in range(X_std.shape[1]):
        mean = np.mean(X_std[:, i])
        std = np.std(X_std[:, i])
        X_std[:, i] = (X_std[:, i] - mean) / std
    return X_std

# Helper function to calculate F1 Score
def f1_score(y, y_hat):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y)):
        if y[i] == 1 and y_hat[i] == 1:
            tp += 1
        elif y[i] == 1 and y_hat[i] == 0:
            fn += 1
        elif y[i] == 0 and y_hat[i] == 1:
            fp += 1
        elif y[i] == 0 and y_hat[i] == 0:
            tn += 1

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    return f1

# The custom Logistic Regression Class (corrected and completed)
class CustomLogisticRegression:
    def sigmoid(self, z):
        return 1 / (1 + exp(-z))

    def initialize(self, X):
        # Add a bias term (column of ones) to X
        X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
        # Initialize weights for features + bias term
        weights = np.zeros((X_with_bias.shape[1], 1))
        return weights, X_with_bias

    def fit(self, X, y, alpha=0.01, iterations=500):
        weights, X_b = self.initialize(X)
        cost_list = np.zeros(iterations)
        y_reshaped = y.reshape(-1, 1) # Ensure y is a column vector

        for i in range(iterations):
            z = dot(X_b, weights)
            predictions = self.sigmoid(z)
            gradient = dot(X_b.T, predictions - y_reshaped) / len(y)
            weights = weights - alpha * gradient

            # Optional: Calculate cost for plotting later
            cost = -np.mean(y_reshaped * log(predictions + 1e-7) + (1 - y_reshaped) * log(1 - predictions + 1e-7))
            cost_list[i] = cost

        self.weights = weights
        return cost_list

    def predict(self, X, threshold=0.5):
        # Add bias term for prediction
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        z = dot(X_b, self.weights)
        predictions = self.sigmoid(z)

        # Return binary predictions based on the threshold
        return [1 if i > threshold else 0 for i in predictions]

# Standardize the data for the custom model
x_train_std = standardize(x_train)
x_test_std = standardize(x_test)

# Train the custom model
custom_model = CustomLogisticRegression()
cost_history = custom_model.fit(x_train_std, y_train)

# Make predictions
y_pred_custom = custom_model.predict(x_test_std)

# Evaluate the custom model
f1_custom = f1_score(y_test, y_pred_custom)
cm_custom = confusion_matrix(y_test, y_pred_custom)
acc_custom = (cm_custom[0, 0] + cm_custom[1, 1]) / np.sum(cm_custom)

print("Custom Model Predictions (first 20):", y_pred_custom[:20])
print("Custom Model F1 Score:", f1_custom)
print("Custom Model Confusion Matrix:\n", cm_custom)
print(f"Custom Model Accuracy: {acc_custom:.4f}")
print("-" * 50)