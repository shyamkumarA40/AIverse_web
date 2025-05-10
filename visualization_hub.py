import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans

st.title("📊 Interactive Data Visualization Hub")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Data:")
    st.dataframe(df)

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    st.sidebar.header("Choose Visualizations")

    # Pairplot
    if st.sidebar.checkbox("Show Pair Plot (Max 5 columns)"):
        selected = st.multiselect("Choose numeric columns for pair plot", num_cols, default=num_cols[:3])
        if len(selected) > 1:
            fig = sns.pairplot(df[selected])
            st.pyplot(fig)

    # Correlation Heatmap
    if st.sidebar.checkbox("Show Correlation Heatmap"):
        fig, ax = plt.subplots()
        sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Histogram
    if st.sidebar.checkbox("Show Histogram"):
        column = st.selectbox("Select column for histogram", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        st.pyplot(fig)

    # Box Plot
    if st.sidebar.checkbox("Show Box Plot"):
        column = st.selectbox("Select column for box plot", num_cols, key="box")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[column], ax=ax)
        st.pyplot(fig)

    # t-SNE / UMAP for Clustering
    if st.sidebar.checkbox("t-SNE or UMAP Clustering Visualization"):
        method = st.radio("Choose dimensionality reduction method", ["t-SNE", "UMAP"])
        n_clusters = st.slider("Number of clusters (KMeans)", min_value=2, max_value=10, value=3)

        # Preprocessing
        df_clean = df[num_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean)

        # Dimensionality reduction
        if method == "t-SNE":
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        else:
            reducer = umap.UMAP(n_components=2, random_state=42)

        X_reduced = reducer.fit_transform(X_scaled)

        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_reduced)

        # Plotting
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='tab10')
        plt.title(f"{method} + KMeans Clustering")
        st.pyplot(fig)

        df_clustered = df.copy()
        df_clustered["Cluster"] = clusters
        st.write("Clustered Data Sample:")
        st.dataframe(df_clustered.head())

else:
    st.info("Please upload a CSV file to begin.")
