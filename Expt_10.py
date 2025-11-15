import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = datasets.load_iris()
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

# Display first few rows
print(df.head())

# Standardize the data
scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Heatmap of scaled feature correlations
plt.figure(figsize=(8, 6))
sns.heatmap(scaled_data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Scaled Features")
plt.show()

# Apply PCA
pca = PCA(n_components=3)
data_pca = pca.fit_transform(scaled_data)
data_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2', 'PC3'])

# Display PCA-transformed data
print(data_pca.head())

# Heatmap of PCA component correlations
plt.figure(figsize=(6, 5))
sns.heatmap(data_pca.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of PCA Components")
plt.show()