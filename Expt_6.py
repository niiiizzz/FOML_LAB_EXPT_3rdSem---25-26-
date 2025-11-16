# -------------------------------------------
#                EXPT - 6
# -------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm

# Load dataset
url = "https://raw.githubusercontent.com/adashofdata/muffin-cupcake/master/recipes_muffins_cupcakes.csv"
recipes = pd.read_csv(url)

# Features and target
X = recipes[['Sugar', 'Flour']].values
y = np.where(recipes['Type'] == 'Muffin', 0, 1)

# ------------------------
# 1. Plot data points only
# ------------------------
plt.figure(figsize=(8,6))
colors = ['red' if label==0 else 'blue' for label in y]
labels = ['Muffin' if label==0 else 'Cupcake' for label in y]
plt.scatter(X[:,0], X[:,1], c=colors, s=70)

# Custom legend
from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], marker='o', color='w', label='Muffin',
                          markerfacecolor='red', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Cupcake',
                          markerfacecolor='blue', markersize=10)]
plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

plt.xlabel('Sugar')
plt.ylabel('Flour')
plt.title("Muffins vs Cupcakes (Data Points Only)")
plt.tight_layout()
plt.show()

# ------------------------
# 2. Plot SVM decision boundary, margins, and support vectors
# ------------------------
# Train linear SVM
model = svm.SVC(kernel='linear')
model.fit(X, y)

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=colors, s=70)

# Get hyperplane and margins
w = model.coef_[0]
b = model.intercept_[0]
x_plot = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
y_plot = -(w[0]/w[1])*x_plot - b/w[1]

margin = 1/np.linalg.norm(w)
y_margin_up = y_plot + np.sqrt(1 + (w[0]/w[1])**2)*margin
y_margin_down = y_plot - np.sqrt(1 + (w[0]/w[1])**2)*margin

plt.plot(x_plot, y_plot, 'k-', label='Decision Boundary')
plt.plot(x_plot, y_margin_up, 'k--', label='Margin')
plt.plot(x_plot, y_margin_down, 'k--')

plt.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s=100,
            facecolors='none', edgecolors='k', label='Support Vectors')

# Custom legend (colors + SVM elements)
legend_elements += [Line2D([0], [0], color='k', lw=2, label='Decision Boundary'),
                    Line2D([0], [0], color='k', lw=2, ls='--', label='Margin'),
                    Line2D([0], [0], marker='o', color='w', label='Support Vectors',
                           markerfacecolor='none', markeredgecolor='k', markersize=10)]
plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

plt.xlabel('Sugar')
plt.ylabel('Flour')
plt.title("SVM Decision Boundary (Muffins vs Cupcakes)")
plt.tight_layout()
plt.show()