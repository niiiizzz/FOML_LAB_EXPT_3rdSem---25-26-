import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load the Iris dataset
iris = load_iris()

# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

plt.figure(figsize=(15, 8))

# Loop through all feature pairs
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    # Select the two corresponding features
    X = iris.data[:, pair]
    y = iris.target

    # Train the Decision Tree classifier
    clf = DecisionTreeClassifier().fit(X, y)

    # Define the mesh grid for plotting
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step),
        np.arange(y_min, y_max, plot_step)
    )

    # Predict on the mesh grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.subplot(2, 3, pairidx + 1)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])

    # Plot training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=iris.target_names[i],
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15
        )

    plt.axis("tight")

plt.suptitle("Decision Surface of Decision Trees Trained on Pairs of Features", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.legend(loc="lower right", borderpad=0.2, handletextpad=0.2)

# Plot the Decision Tree trained on all features
plt.figure(figsize=(12, 8))
clf_full = DecisionTreeClassifier().fit(iris.data, iris.target)
plot_tree(clf_full, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree Trained on All Iris Features")
plt.show()
