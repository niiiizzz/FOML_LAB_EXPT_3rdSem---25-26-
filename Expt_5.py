# -----------------------------------------
#            EXPT - 5
# ----------------------------------------
import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# ----------------------------
# 1. Download / load dataset
# ----------------------------
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
LOCAL_FILE = "data_banknote_authentication.txt"

# Download the dataset if not already present
if not os.path.exists(LOCAL_FILE):
    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, LOCAL_FILE)
    print("Downloaded to", LOCAL_FILE)

# Column names as per UCI repository
cols = ['variance', 'skewness', 'curtosis', 'entropy', 'class']

# Read into pandas
bnotes = pd.read_csv(LOCAL_FILE, header=None, names=cols)

print("\nFirst 10 rows of the dataset:")
print(bnotes.head(10))
print("\nDataset shape:", bnotes.shape)
print("\nColumn names:", bnotes.columns.tolist())

# ----------------------------
# 2. Prepare features and target
# ----------------------------
X = bnotes.drop('class', axis=1)
y = bnotes['class']

# For label consistency, ensure y is integer
y = y.astype(int)

# ----------------------------
# 3. Function to train & evaluate
# ----------------------------
def run_mlp(X_train, X_test, y_train, y_test, activation):
    print("\n" + "="*40)
    print("Activation:", activation)
    print("="*40)

    mlp = MLPClassifier(max_iter=500, activation=activation, random_state=42)
    mlp.fit(X_train, y_train)
    pred = mlp.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("Classification Report:")
    print(classification_report(y_test, pred))

# ----------------------------
# 4. First split: 80-20
# ----------------------------
print("\n--- Train-test split 80/20 ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for act in ['relu', 'logistic', 'tanh', 'identity']:
    run_mlp(X_train, X_test, y_train, y_test, act)

# ----------------------------
# 5. Second split: 70-30
# ----------------------------
print("\n--- Train-test split 70/30 ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for act in ['relu', 'logistic', 'tanh', 'identity']:
    run_mlp(X_train, X_test, y_train, y_test, act)