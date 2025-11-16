
# -------‐-------------------
#           expt 2
# ---------------------------
# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Try loading dataset from local CSV; otherwise load from seaborn
try:
    df = pd.read_csv('iris.csv')
    print("✅ Loaded dataset from local iris.csv file.")
except FileNotFoundError:
    print("⚠️ Local iris.csv not found. Loading built-in seaborn Iris dataset instead.")
    df = sns.load_dataset('iris')
    df.rename(columns={'species': 'variety'}, inplace=True)  # rename for consistency

# Display basic info
print(df.head())
print("Shape of dataset:", df.shape)

# --- UNIVARIATE ANALYSIS ---

# Create subsets by flower variety
df_Setosa = df[df['variety'] == 'setosa']
df_Virginica = df[df['variety'] == 'virginica']
df_Versicolor = df[df['variety'] == 'versicolor']

# Univariate for Sepal Width
plt.scatter(df_Setosa['sepal_width'], np.zeros_like(df_Setosa['sepal_width']), label='Setosa')
plt.scatter(df_Virginica['sepal_width'], np.zeros_like(df_Virginica['sepal_width']), label='Virginica')
plt.scatter(df_Versicolor['sepal_width'], np.zeros_like(df_Versicolor['sepal_width']), label='Versicolor')
plt.xlabel('Sepal Width')
plt.legend()
plt.title('Univariate Analysis: Sepal Width')
plt.show()

# Univariate for Sepal Length
plt.scatter(df_Setosa['sepal_length'], np.zeros_like(df_Setosa['sepal_length']), label='Setosa')
plt.scatter(df_Virginica['sepal_length'], np.zeros_like(df_Virginica['sepal_length']), label='Virginica')
plt.scatter(df_Versicolor['sepal_length'], np.zeros_like(df_Versicolor['sepal_length']), label='Versicolor')
plt.xlabel('Sepal Length')
plt.legend()
plt.title('Univariate Analysis: Sepal Length')
plt.show()

# Univariate for Petal Width
plt.scatter(df_Setosa['petal_width'], np.zeros_like(df_Setosa['petal_width']), label='Setosa')
plt.scatter(df_Virginica['petal_width'], np.zeros_like(df_Virginica['petal_width']), label='Virginica')
plt.scatter(df_Versicolor['petal_width'], np.zeros_like(df_Versicolor['petal_width']), label='Versicolor')
plt.xlabel('Petal Width')
plt.legend()
plt.title('Univariate Analysis: Petal Width')
plt.show()

# Univariate for Petal Length
plt.scatter(df_Setosa['petal_length'], np.zeros_like(df_Setosa['petal_length']), label='Setosa')
plt.scatter(df_Virginica['petal_length'], np.zeros_like(df_Virginica['petal_length']), label='Virginica')
plt.scatter(df_Versicolor['petal_length'], np.zeros_like(df_Versicolor['petal_length']), label='Versicolor')
plt.xlabel('Petal Length')
plt.legend()
plt.title('Univariate Analysis: Petal Length')
plt.show()

# --- BIVARIATE ANALYSIS ---

# Sepal Width vs Petal Width
sns.FacetGrid(df, hue='variety', height=5).map(plt.scatter, 'sepal_width', 'petal_width').add_legend()
plt.title('Bivariate Analysis: Sepal Width vs Petal Width')
plt.show()

# Sepal Length vs Petal Length
sns.FacetGrid(df, hue='variety', height=5).map(plt.scatter, 'sepal_length', 'petal_length').add_legend()
plt.title('Bivariate Analysis: Sepal Length vs Petal Length')
plt.show()

# --- MULTIVARIATE ANALYSIS ---

sns.pairplot(df, hue='variety', height=2)
plt.suptitle('Multivariate Analysis (All Features)', y=1.02)
plt.show()