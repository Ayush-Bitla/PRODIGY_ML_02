import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

st.set_page_config(page_title="🧠 Use KMeans Model", layout='wide')
st.title("🔍 Predict Customer Segments Using Saved K-Means Model")

# Upload dataset and model
uploaded_file = st.file_uploader("📤 Upload your customer CSV file", type=["csv"])
model_file = st.file_uploader("📦 Upload trained KMeans model (.pkl)", type=["pkl"])

if uploaded_file and model_file:
    try:
        df = pd.read_csv(uploaded_file)
        model = joblib.load(model_file)
    except Exception as e:
        st.error(f"❌ Error loading file or model: {e}")
        st.stop()

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Detect and preprocess if raw (Online Retail format)
    if 'Invoice' in df.columns and 'Customer ID' in df.columns:
        st.warning("⚠️ Detected raw Online Retail data. Running RFM preprocessing...")
        df.dropna(subset=['Customer ID'], inplace=True)
        df = df[df['Quantity'] > 0]
        df['TotalPrice'] = df['Quantity'] * df['Price']
        reference_date = df['InvoiceDate'].max()
        rfm = df.groupby('Customer ID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,
            'Invoice': 'nunique',
            'TotalPrice': 'sum'
        }).reset_index()
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        df = rfm.drop('CustomerID', axis=1)
    else:
        # Preprocessing for Mall Customers-style dataset
        if 'CustomerID' in df.columns:
            df = df.drop('CustomerID', axis=1)
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].replace({'Male': 0, 'Female': 1})

    st.subheader("🔄 Preprocessed Data")
    st.dataframe(df.head())

    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Predict
    try:
        labels = model.predict(scaled_data)
        df['Predicted Cluster'] = labels
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    st.subheader("✅ Predicted Clusters")
    st.dataframe(df.head())

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(scaled_data)

    fig, ax = plt.subplots()
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set2', s=100, ax=ax)
    ax.set_title("📌 Cluster Visualization (PCA)")
    st.pyplot(fig)

    st.subheader("📊 Cluster Summary (Averages)")
    st.dataframe(df.groupby('Predicted Cluster').mean())

else:
    st.info("👆 Please upload both a dataset and a trained `.pkl` model to begin.")
