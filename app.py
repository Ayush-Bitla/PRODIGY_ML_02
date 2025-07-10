import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(layout='wide')
st.title("🛍️ Customer Segmentation App")
st.markdown("This app applies **K-Means** and **DBSCAN** clustering on uploaded customer data.")

# File upload
uploaded_file = st.file_uploader("📤 Upload your dataset (.csv)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to read the CSV file: {e}")
        st.stop()

    # Optional cleanup
    df = df.drop(columns=['CustomerID'], errors='ignore')

    # Handle gender if it exists
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'Male': 0, 'Female': 1})
    
    # Keep only numeric columns
    data = df.select_dtypes(include=np.number)

    if data.shape[1] < 2:
        st.warning("⚠️ Please upload a dataset with at least 2 numeric features.")
        st.stop()

    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    st.subheader("📊 Descriptive Statistics")
    st.write(data.describe())

    # Scale
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(scaled_data)

    # K-Means
    st.sidebar.title("⚙️ K-Means Parameters")
    k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 5)

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(scaled_data)
    silhouette_kmeans = silhouette_score(scaled_data, kmeans_labels)

    joblib.dump(kmeans, "kmeans_model.pkl")

    st.sidebar.markdown(f"**K-Means Silhouette Score:** `{silhouette_kmeans:.3f}`")

    st.subheader("📌 K-Means Clustering Results")
    kmeans_df = data.copy()
    kmeans_df['Cluster'] = kmeans_labels

    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='Set2', s=100, ax=ax1)
    ax1.set_title("K-Means Clusters (PCA Reduced)")
    st.pyplot(fig1)

    st.write("📈 Cluster Averages (K-Means):")
    st.dataframe(kmeans_df.groupby('Cluster').mean())

    # Download model
    with open("kmeans_model.pkl", "rb") as file:
        st.download_button("⬇️ Download Trained KMeans Model", file, "kmeans_model.pkl")

    # DBSCAN
    st.sidebar.title("⚙️ DBSCAN Parameters")
    eps = st.sidebar.slider("eps (radius)", 0.1, 2.0, 0.4, 0.1)

    dbscan = DBSCAN(eps=eps, min_samples=5)
    db_labels = dbscan.fit_predict(scaled_data)

    mask = db_labels != -1
    if len(set(db_labels)) > 1 and mask.sum() > 0:
        silhouette_db = silhouette_score(scaled_data[mask], db_labels[mask])
    else:
        silhouette_db = -1

    st.sidebar.markdown(f"**DBSCAN Silhouette Score:** `{silhouette_db:.3f}`")

    st.subheader("📌 DBSCAN Clustering Results")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=db_labels, palette='tab10', s=100, ax=ax2)
    ax2.set_title(f"DBSCAN Clusters (PCA Reduced) [eps={eps}]")
    st.pyplot(fig2)

    df['DBSCAN_Cluster'] = db_labels
    st.write("📊 DBSCAN Cluster Count:")
    st.write(pd.Series(db_labels).value_counts())

    # Comparison
    if silhouette_db > silhouette_kmeans:
        st.success(f"✅ DBSCAN performed better (Score: {silhouette_db:.3f}) than K-Means (Score: {silhouette_kmeans:.3f})")
    else:
        st.info(f"ℹ️ K-Means performed better (Score: {silhouette_kmeans:.3f}) than DBSCAN (Score: {silhouette_db:.3f})")

    # Comparison table
    st.subheader("🔄 Comparison Table (First 10 Rows)")
    st.dataframe(pd.DataFrame({
        "KMeans Cluster": kmeans_labels,
        "DBSCAN Cluster": db_labels
    }).head(10))

else:
    st.info("👆 Please upload a dataset to get started.")
