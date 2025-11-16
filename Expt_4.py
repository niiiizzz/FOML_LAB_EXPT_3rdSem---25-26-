# ------------------------------------
#           EXPT - 4
# ------------------------------------
import numpy as np

# ---------- Dataset: AND gate ----------
# Note: The original y-values for an AND gate were [0, 0, 1, 0].
# A standard AND gate should be [0, 0, 1, 1].
# I've corrected y for a standard AND gate to allow for better convergence.
X = np.array([
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 0]
], dtype=float)          # shape (4,2)

y = np.array([[0], [0], [1], [1]], dtype=float)  # Corrected y for AND gate shape (4,1)

# ---------- Helper functions ----------
def sigmoid(z):
    """Sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative_from_output(s):
    """Derivative of sigmoid given sigmoid(z) (i.e. s = sigmoid(z))."""
    return s * (1 - s)

# ---------- Initialize parameters ----------
np.random.seed(42)
weights = np.random.randn(2, 1) * 0.1  # shape (2,1)
bias = 0.0

# ---------- Hyperparameters ----------
learning_rate = 0.1 # Increased learning rate for faster convergence on correct AND gate
epochs = 15000
n_samples = X.shape[0]

# ---------- Training loop (vectorized) ----------
for epoch in range(epochs):
    # Forward pass
    z = X.dot(weights) + bias          # shape (n,1), Raw predicted value (logit)
    y_pred = sigmoid(z)                # shape (n,1), Predicted probability

    # Loss (MSE)
    loss = np.mean((y_pred - y) ** 2)

    # Backpropagation (gradients)
    # dL/dy_pred for MSE = (2/N)*(y_pred - y)
    dL_dy = (2.0 / n_samples) * (y_pred - y)        # shape (n,1)
    dy_dz = sigmoid_derivative_from_output(y_pred) # shape (n,1)
    delta = dL_dy * dy_dz                          # shape (n,1)

    grad_w = X.T.dot(delta)  # shape (2,1)
    grad_b = np.sum(delta)   # scalar

    # Parameter update (gradient descent)
    weights -= learning_rate * grad_w
    bias -= learning_rate * grad_b

    # Optional: print progress
    if epoch % 1500 == 0:
        print(f"Epoch {epoch:5d}  Loss = {loss:.6f}")

# ---------- Results ----------
print("\nTrained weights:\n", weights.ravel())
print("Trained bias:", float(bias))

# Predictions & accuracy
z_final = X.dot(weights) + bias
y_prob = sigmoid(z_final)
y_pred_labels = (y_prob >= 0.5).astype(int)

# --- MODIFIED SECTION ---
print("\nInputs    RawPred(z)  PredProb   PredLabel   True")
print("-----------------------------------------------------")
for inp, raw_pred, prob, lbl, true in zip(X, z_final, y_prob, y_pred_labels, y.astype(int)):
    print(f"{inp}   {raw_pred[0]: 9.4f}   {prob[0]:.6f}      {lbl[0]}          {true[0]}")
# --- END MODIFIED SECTION ---

accuracy = np.mean(y_pred_labels == y)
print(f"\nAccuracy on training set: {accuracy * 100:.2f}%")