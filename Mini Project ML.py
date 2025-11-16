"""
CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING
Dataset: Mall Customers (Kaggle)
Purpose: Segment customers based on demographics and spending behavior
"""

import warnings

warnings.filterwarnings('ignore')

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING")
print("="*80)

# ----------------------
# 1. Load Dataset with Multiple Fallback URLs
# ----------------------
print("\n[1/12] Loading Mall Customers Dataset...")

dataset_urls = [
    "https://gist.githubusercontent.com/ryanorsinger/cc276eea59e8295204d1f581c8da509f/raw/d47b2f1a1c4ba57be8f8b67f3ac99a3f8b7c9d95/mall_customers.csv",
    "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv",
    "https://raw.githubusercontent.com/rapsingh/MALL-CUSTOMERS-SEGMENTATION/main/Mall_Customers.csv",
    "https://raw.githubusercontent.com/tanishq21/Mall-Customers/main/Mall_Customers.csv"
]

df = None
for url in dataset_urls:
    try:
        print(f"Trying: {url[:60]}...")
        df = pd.read_csv(url)
        print(f"✓ Successfully loaded!")
        break
    except Exception as e:
        print(f"✗ Failed: {e}")
        continue

if df is None:
    print("\n" + "="*80)
    print("ERROR: Could not download dataset from any source.")
    print("="*80)
    print("\nPlease download manually:")
    print("1. Visit: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python")
    print("2. Download 'Mall_Customers.csv'")
    print("3. Place it in the same directory and run: df = pd.read_csv('Mall_Customers.csv')")
    raise SystemExit("Dataset not available")

print(f"\n✓ Dataset Shape: {df.shape}")
print(f"✓ Columns: {list(df.columns)}")

# ----------------------
# 2. Data Exploration
# ----------------------
print("\n[2/12] Exploring Dataset...")
print("\nFirst 5 rows:")
print(df.head())

print("\n" + "-"*80)
print("Dataset Information:")
print(df.info())

print("\n" + "-"*80)
print("Statistical Summary:")
print(df.describe())

print("\n" + "-"*80)
print("Missing Values:")
print(df.isnull().sum())

print("\n" + "-"*80)
print("Gender Distribution:")
print(df['Gender'].value_counts())

# ----------------------
# 3. Data Preprocessing
# ----------------------
print("\n[3/12] Preprocessing Data...")

# Create a copy for processing
data = df.copy()

# Standardize column names
data.columns = data.columns.str.strip()

# Encode Gender
le = LabelEncoder()
data['Gender_Encoded'] = le.fit_transform(data['Gender'])
print(f"✓ Gender encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Select features for clustering
feature_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[feature_cols].values

print(f"✓ Features selected: {feature_cols}")
print(f"✓ Feature matrix shape: {X.shape}")

# ----------------------
# 4. Feature Scaling
# ----------------------
print("\n[4/12] Scaling Features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✓ Features normalized using StandardScaler")

# ----------------------
# 5. Elbow Method - Finding Optimal K
# ----------------------
print("\n[5/12] Finding Optimal Number of Clusters (Elbow Method)...")

inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    print(f"   K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.4f}")

# Plot Elbow Curve
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Elbow plot
axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
axes[0].set_title('Elbow Method For Optimal K', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=5, color='r', linestyle='--', alpha=0.7, label='Optimal K=5')
axes[0].legend()

# Silhouette plot
axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score For Different K', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=5, color='r', linestyle='--', alpha=0.7, label='Optimal K=5')
axes[1].legend()

plt.tight_layout()
plt.savefig('elbow_method.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: elbow_method.png")
plt.show()

# ----------------------
# 6. Train K-Means Model with Optimal K
# ----------------------
print("\n[6/12] Training K-Means Model...")
optimal_k = 5
print(f"✓ Using K = {optimal_k} clusters")

kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
data['Cluster'] = clusters

print(f"\n✓ Model trained successfully!")
print(f"✓ Cluster distribution:")
print(data['Cluster'].value_counts().sort_index())

# ----------------------
# 7. Model Evaluation Metrics
# ----------------------
print("\n[7/12] Evaluating Model Performance...")

silhouette = silhouette_score(X_scaled, clusters)
davies_bouldin = davies_bouldin_score(X_scaled, clusters)
calinski_harabasz = calinski_harabasz_score(X_scaled, clusters)

print(f"\n📊 Clustering Performance Metrics:")
print(f"   Silhouette Score:       {silhouette:.4f} (Higher is better, range: -1 to 1)")
print(f"   Davies-Bouldin Index:   {davies_bouldin:.4f} (Lower is better)")
print(f"   Calinski-Harabasz Score: {calinski_harabasz:.2f} (Higher is better)")

if silhouette > 0.5:
    print(f"\n   ✓ EXCELLENT: Silhouette score > 0.5 indicates well-separated clusters!")
elif silhouette > 0.3:
    print(f"\n   ✓ GOOD: Silhouette score > 0.3 indicates reasonable cluster structure!")
else:
    print(f"\n   ⚠ FAIR: Consider adjusting number of clusters")

# ----------------------
# 8. Cluster Analysis
# ----------------------
print("\n[8/12] Analyzing Cluster Characteristics...")

# Calculate cluster statistics
cluster_summary = data.groupby('Cluster')[feature_cols].mean()
print("\n" + "="*80)
print("CLUSTER PROFILES (Average Values)")
print("="*80)
print(cluster_summary)

# Cluster sizes
cluster_sizes = data['Cluster'].value_counts().sort_index()
print("\n" + "-"*80)
print("Cluster Sizes:")
for i in range(optimal_k):
    print(f"   Cluster {i}: {cluster_sizes[i]} customers ({cluster_sizes[i]/len(data)*100:.1f}%)")

# ----------------------
# 9. Cluster Naming & Interpretation
# ----------------------
print("\n[9/12] Naming Customer Segments...")

cluster_names = {
    0: "High Spenders - High Income",
    1: "Low Spenders - Low Income", 
    2: "Average Customers",
    3: "High Spenders - Low Income",
    4: "Low Spenders - High Income"
}

# Assign names based on actual cluster characteristics
cluster_means = data.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()

def name_cluster(row):
    income = row['Annual Income (k$)']
    spending = row['Spending Score (1-100)']
    
    if income > 60 and spending > 60:
        return "Target Group (High Income, High Spending)"
    elif income < 40 and spending < 40:
        return "Low Priority (Low Income, Low Spending)"
    elif income > 60 and spending < 40:
        return "Potential Targets (High Income, Low Spending)"
    elif income < 40 and spending > 60:
        return "Careful Spenders (Low Income, High Spending)"
    else:
        return "Standard Customers (Average Profile)"

cluster_names_actual = {}
for cluster_id in range(optimal_k):
    row = cluster_means.loc[cluster_id]
    cluster_names_actual[cluster_id] = name_cluster(row)

data['Cluster_Name'] = data['Cluster'].map(cluster_names_actual)

print("\n" + "="*80)
print("CUSTOMER SEGMENTS:")
print("="*80)
for cluster_id, name in cluster_names_actual.items():
    count = cluster_sizes[cluster_id]
    pct = count/len(data)*100
    avg_income = cluster_summary.loc[cluster_id, 'Annual Income (k$)']
    avg_spending = cluster_summary.loc[cluster_id, 'Spending Score (1-100)']
    avg_age = cluster_summary.loc[cluster_id, 'Age']
    
    print(f"\n📍 Cluster {cluster_id}: {name}")
    print(f"   Size: {count} customers ({pct:.1f}%)")
    print(f"   Avg Age: {avg_age:.1f} years")
    print(f"   Avg Income: ${avg_income:.1f}k")
    print(f"   Avg Spending Score: {avg_spending:.1f}/100")

# ----------------------
# 10. Visualizations
# ----------------------
print("\n[10/12] Creating Visualizations...")

# Set color palette
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

# Visualization 1: 2D Scatter - Income vs Spending
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Income vs Spending Score
for i in range(optimal_k):
    cluster_data = data[data['Cluster'] == i]
    axes[0, 0].scatter(
        cluster_data['Annual Income (k$)'],
        cluster_data['Spending Score (1-100)'],
        c=colors[i],
        label=f'Cluster {i}',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5,
        s=100
    )

# Plot centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
axes[0, 0].scatter(
    centroids[:, 1],
    centroids[:, 2],
    c='red',
    s=300,
    marker='*',
    edgecolors='black',
    linewidth=2,
    label='Centroids'
)

axes[0, 0].set_xlabel('Annual Income (k$)', fontsize=12)
axes[0, 0].set_ylabel('Spending Score (1-100)', fontsize=12)
axes[0, 0].set_title('Customer Segments: Income vs Spending', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Age vs Spending Score
for i in range(optimal_k):
    cluster_data = data[data['Cluster'] == i]
    axes[0, 1].scatter(
        cluster_data['Age'],
        cluster_data['Spending Score (1-100)'],
        c=colors[i],
        label=f'Cluster {i}',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5,
        s=100
    )

axes[0, 1].scatter(
    centroids[:, 0],
    centroids[:, 2],
    c='red',
    s=300,
    marker='*',
    edgecolors='black',
    linewidth=2,
    label='Centroids'
)

axes[0, 1].set_xlabel('Age', fontsize=12)
axes[0, 1].set_ylabel('Spending Score (1-100)', fontsize=12)
axes[0, 1].set_title('Customer Segments: Age vs Spending', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Age vs Income
for i in range(optimal_k):
    cluster_data = data[data['Cluster'] == i]
    axes[1, 0].scatter(
        cluster_data['Age'],
        cluster_data['Annual Income (k$)'],
        c=colors[i],
        label=f'Cluster {i}',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5,
        s=100
    )

axes[1, 0].scatter(
    centroids[:, 0],
    centroids[:, 1],
    c='red',
    s=300,
    marker='*',
    edgecolors='black',
    linewidth=2,
    label='Centroids'
)

axes[1, 0].set_xlabel('Age', fontsize=12)
axes[1, 0].set_ylabel('Annual Income (k$)', fontsize=12)
axes[1, 0].set_title('Customer Segments: Age vs Income', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Cluster Distribution
cluster_counts = data['Cluster'].value_counts().sort_index()
axes[1, 1].bar(range(optimal_k), cluster_counts.values, color=colors, edgecolor='black', linewidth=1.5)
axes[1, 1].set_xlabel('Cluster', fontsize=12)
axes[1, 1].set_ylabel('Number of Customers', fontsize=12)
axes[1, 1].set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(range(optimal_k))
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, v in enumerate(cluster_counts.values):
    axes[1, 1].text(i, v + 1, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('customer_segments_2d.png', dpi=150, bbox_inches='tight')
print("✓ Saved: customer_segments_2d.png")
plt.show()

# Visualization 2: 3D Scatter Plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

for i in range(optimal_k):
    cluster_data = data[data['Cluster'] == i]
    ax.scatter(
        cluster_data['Age'],
        cluster_data['Annual Income (k$)'],
        cluster_data['Spending Score (1-100)'],
        c=colors[i],
        label=f'Cluster {i}',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5,
        s=60
    )

# Plot centroids
ax.scatter(
    centroids[:, 0],
    centroids[:, 1],
    centroids[:, 2],
    c='red',
    s=500,
    marker='*',
    edgecolors='black',
    linewidth=2,
    label='Centroids'
)

ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Annual Income (k$)', fontsize=11)
ax.set_zlabel('Spending Score', fontsize=11)
ax.set_title('3D Customer Segmentation', fontsize=14, fontweight='bold')
ax.legend()

plt.savefig('customer_segments_3d.png', dpi=150, bbox_inches='tight')
print("✓ Saved: customer_segments_3d.png")
plt.show()

# Visualization 3: Box plots by cluster
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

data.boxplot(column='Age', by='Cluster', ax=axes[0], patch_artist=True)
axes[0].set_title('Age Distribution by Cluster', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Cluster', fontsize=11)
axes[0].set_ylabel('Age', fontsize=11)

data.boxplot(column='Annual Income (k$)', by='Cluster', ax=axes[1], patch_artist=True)
axes[1].set_title('Income Distribution by Cluster', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Cluster', fontsize=11)
axes[1].set_ylabel('Annual Income (k$)', fontsize=11)

data.boxplot(column='Spending Score (1-100)', by='Cluster', ax=axes[2], patch_artist=True)
axes[2].set_title('Spending Score Distribution by Cluster', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Cluster', fontsize=11)
axes[2].set_ylabel('Spending Score', fontsize=11)

plt.suptitle('')
plt.tight_layout()
plt.savefig('cluster_distributions.png', dpi=150, bbox_inches='tight')
print("✓ Saved: cluster_distributions.png")
plt.show()

# Visualization 4: Heatmap of cluster characteristics
cluster_heatmap_data = data.groupby('Cluster')[feature_cols].mean()

plt.figure(figsize=(10, 6))
sns.heatmap(
    cluster_heatmap_data.T,
    annot=True,
    fmt='.2f',
    cmap='YlOrRd',
    linewidths=0.5,
    cbar_kws={'label': 'Average Value'}
)
plt.title('Cluster Characteristics Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig('cluster_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ Saved: cluster_heatmap.png")
plt.show()

# ----------------------
# 11. Save Model and Results
# ----------------------
print("\n[11/12] Saving Model and Results...")

# Save model
joblib.dump(kmeans, 'kmeans_model.pkl')
print("✓ Saved: kmeans_model.pkl")

# Save scaler
joblib.dump(scaler, 'scaler.pkl')
print("✓ Saved: scaler.pkl")

# Save label encoder
joblib.dump(le, 'label_encoder.pkl')
print("✓ Saved: label_encoder.pkl")

# Save results to CSV
output_df = data[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 
                   'Spending Score (1-100)', 'Cluster', 'Cluster_Name']]
output_df.to_csv('customer_segments.csv', index=False)
print("✓ Saved: customer_segments.csv")

# Save cluster summary
cluster_summary_full = data.groupby('Cluster').agg({
    'CustomerID': 'count',
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'Gender': lambda x: x.mode()[0]
})
cluster_summary_full.columns = ['Count', 'Avg_Age', 'Avg_Income', 'Avg_Spending', 'Dominant_Gender']
cluster_summary_full.to_csv('cluster_summary.csv')
print("✓ Saved: cluster_summary.csv")

# Save metrics
with open('clustering_metrics.txt', 'w') as f:
    f.write("CUSTOMER SEGMENTATION - K-MEANS CLUSTERING\n")
    f.write("="*60 + "\n\n")
    f.write(f"Dataset: Mall Customers\n")
    f.write(f"Total Customers: {len(data)}\n")
    f.write(f"Number of Clusters: {optimal_k}\n\n")
    f.write("PERFORMANCE METRICS:\n")
    f.write("-"*60 + "\n")
    f.write(f"Silhouette Score:         {silhouette:.4f}\n")
    f.write(f"Davies-Bouldin Index:     {davies_bouldin:.4f}\n")
    f.write(f"Calinski-Harabasz Score:  {calinski_harabasz:.2f}\n\n")
    f.write("CLUSTER SUMMARY:\n")
    f.write("-"*60 + "\n")
    f.write(cluster_summary_full.to_string())
print("✓ Saved: clustering_metrics.txt")

# ----------------------
# 12. Prediction Function
# ----------------------
print("\n[12/12] Creating Prediction Function...")

def predict_customer_segment(age, annual_income, spending_score):
    """
    Predict which customer segment a new customer belongs to
    
    Parameters:
    - age: Customer age
    - annual_income: Annual income in thousands ($)
    - spending_score: Spending score (1-100)
    
    Returns:
    - cluster_id: Cluster number
    - cluster_name: Cluster description
    """
    # Create feature array
    features = np.array([[age, annual_income, spending_score]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict cluster
    cluster = kmeans.predict(features_scaled)[0]
    cluster_name = cluster_names_actual[cluster]
    
    return cluster, cluster_name

print("✓ Prediction function ready")

# ----------------------
# Test Predictions
# ----------------------
print("\n" + "="*80)
print("SAMPLE PREDICTIONS")
print("="*80)

test_customers = [
    {'age': 25, 'income': 70, 'spending': 80},
    {'age': 45, 'income': 30, 'spending': 20},
    {'age': 35, 'income': 90, 'spending': 30},
    {'age': 22, 'income': 25, 'spending': 75},
    {'age': 50, 'income': 60, 'spending': 50}
]

print(f"\n{'Age':>4s} | {'Income':>8s} | {'Spending':>8s} | {'Cluster':>7s} | {'Segment'}")
print("-"*80)

for customer in test_customers:
    cluster_id, segment_name = predict_customer_segment(
        customer['age'],
        customer['income'],
        customer['spending']
    )
    print(f"{customer['age']:>4d} | ${customer['income']:>7d}k | {customer['spending']:>8d} | Cluster {cluster_id} | {segment_name}")

# ----------------------
# Final Summary
# ----------------------
print("\n" + "="*80)
print("PROJECT COMPLETE! 🎉")
print("="*80)

print(f"\n📊 Clustering Summary:")
print(f"   Total Customers Analyzed: {len(data)}")
print(f"   Number of Segments: {optimal_k}")
print(f"   Features Used: {', '.join(feature_cols)}")

print(f"\n🎯 Model Performance:")
print(f"   Silhouette Score: {silhouette:.4f}")
if silhouette > 0.5:
    print(f"   Rating: ⭐⭐⭐⭐⭐ EXCELLENT")
elif silhouette > 0.4:
    print(f"   Rating: ⭐⭐⭐⭐ VERY GOOD")
elif silhouette > 0.3:
    print(f"   Rating: ⭐⭐⭐ GOOD")

print(f"\n📁 Generated Files:")
files = [
    'kmeans_model.pkl',
    'scaler.pkl', 
    'label_encoder.pkl',
    'customer_segments.csv',
    'cluster_summary.csv',
    'clustering_metrics.txt',
    'elbow_method.png',
    'customer_segments_2d.png',
    'customer_segments_3d.png',
    'cluster_distributions.png',
    'cluster_heatmap.png'
]

for file in files:
    print(f"   • {file}")

print("\n💡 Key Insights:")
for cluster_id in range(optimal_k):
    name = cluster_names_actual[cluster_id]
    count = cluster_sizes[cluster_id]
    pct = count/len(data)*100
    print(f"   • Cluster {cluster_id} ({name}): {count} customers ({pct:.1f}%)")

print("\n" + "="*80)
print("Ready for your mini project presentation! 🚀")
print("="*80)