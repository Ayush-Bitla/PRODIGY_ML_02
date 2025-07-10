import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout='wide')
st.title("🛍️ Customer Segmentation using Clustering")

# Upload file
uploaded_file = st.file_uploader("Upload your dataset (Mall_Customers.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Data")
    st.dataframe(df.head())

    # Preprocess
    data = df.drop("CustomerID", axis=1)
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

    st.subheader("📊 Descriptive Statistics")
    st.write(data.describe())

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # PCA for 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(scaled_data)

    # Sidebar controls
    st.sidebar.title("⚙️ K-Means Parameters")
    k = st.sidebar.slider("Number of clusters (K)", 2, 10, 5)

    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(scaled_data)
    kmeans_score = silhouette_score(scaled_data, kmeans_labels)
    st.sidebar.markdown(f"**K-Means Silhouette Score:** `{kmeans_score:.3f}`")

    st.subheader("📌 K-Means Clustering Results")
    kmeans_df = data.copy()
    kmeans_df['Cluster'] = kmeans_labels

    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='Set2', s=100, ax=ax1)
    ax1.set_title("K-Means Clusters (PCA Reduced)")
    st.pyplot(fig1)

    st.write("Cluster Averages:")
    st.dataframe(kmeans_df.groupby('Cluster').mean())

    # DBSCAN
    st.sidebar.title("⚙️ DBSCAN Parameters")
    eps = st.sidebar.slider("eps (radius)", 0.1, 2.0, 0.4, 0.1)

    dbscan = DBSCAN(eps=eps, min_samples=5)
    db_labels = dbscan.fit_predict(scaled_data)

    mask = db_labels != -1
    if len(set(db_labels)) > 1 and mask.sum() > 0:
        db_score = silhouette_score(scaled_data[mask], db_labels[mask])
    else:
        db_score = -1

    st.sidebar.markdown(f"**DBSCAN Silhouette Score:** `{db_score:.3f}`")

    st.subheader("📌 DBSCAN Clustering Results")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=db_labels, palette='tab10', s=100, ax=ax2)
    ax2.set_title(f"DBSCAN Clusters (PCA Reduced) [eps={eps}]")
    st.pyplot(fig2)

    df['DBSCAN_Cluster'] = db_labels
    st.write("Cluster Count:")
    st.write(pd.Series(db_labels).value_counts())

    if db_score > kmeans_score:
        st.success(f"✅ DBSCAN is better (Score: {db_score:.3f}) than K-Means (Score: {kmeans_score:.3f})!")
    else:
        st.info(f"ℹ️ K-Means (Score: {kmeans_score:.3f}) performed better than DBSCAN (Score: {db_score:.3f})")
