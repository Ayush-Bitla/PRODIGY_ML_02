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
st.markdown("This app applies **K-Means** and **DBSCAN** clustering on mall customer data.")

# File upload
uploaded_file = st.file_uploader("📤 Upload your dataset (e.g., Mall_Customers.csv)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        assert 'CustomerID' in df.columns and 'Gender' in df.columns
    except:
        st.error("❌ The uploaded file does not match expected format.")
        st.stop()

    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    # Preprocess
    data = df.drop("CustomerID", axis=1)
    # Encode Gender column safely
    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].replace({'Male': 0, 'Female': 1})

    st.subheader("📊 Descriptive Statistics")
    st.write(data.describe())

    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(scaled_data)

    # K-Means parameters
    st.sidebar.title("⚙️ K-Means Parameters")
    k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 5)

    # K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(scaled_data)
    joblib.dump(kmeans, "kmeans_model.pkl")
    kmeans_score = silhouette_score(scaled_data, kmeans_labels)
    st.sidebar.markdown(f"**K-Means Silhouette Score:** `{kmeans_score:.3f}`")

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

    # DBSCAN parameters
    st.sidebar.title("⚙️ DBSCAN Parameters")
    eps = st.sidebar.slider("eps (radius)", 0.1, 2.0, 0.4, 0.1)

    # DBSCAN clustering
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
    st.write("📊 DBSCAN Cluster Count:")
    st.write(pd.Series(db_labels).value_counts())

    # Compare
    if db_score > kmeans_score:
        st.success(f"✅ DBSCAN is better (Score: {db_score:.3f}) than K-Means (Score: {kmeans_score:.3f})!")
    else:
        st.info(f"ℹ️ K-Means (Score: {kmeans_score:.3f}) performed better than DBSCAN (Score: {db_score:.3f})")

    # Optional: Combined summary
    st.subheader("🔄 Comparison Table (First 10 Rows)")
    comparison_df = pd.DataFrame({
        "KMeans Cluster": kmeans_labels,
        "DBSCAN Cluster": db_labels
    })
    st.dataframe(comparison_df.head(10))
