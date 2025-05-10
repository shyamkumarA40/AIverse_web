import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Session state setup
if "data" not in st.session_state:
    st.session_state.data = None

st.set_page_config(page_title="AIverse CSV App")
st.title("🤖 AIverse: Unified CSV AI Platform")
import os
os.environ["OPENAI_API_KEY"] = "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"

# === CUSTOM STYLING ===
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');
html, body, .stApp {
    background: linear-gradient(to right, #fddb92, #d1fdff);
    background-attachment: fixed;
    background-size: cover;
    color: #2c3e50 !important;
    font-family: 'Segoe UI', sans-serif !important;
}
.highlight {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 2rem;
}
.highlight h1 {
    font-size: 2.6rem;
    margin: 0;
}
.data-box {
    background-color: #ffffff;
    padding: 1.5rem;
    border: 2px dashed #1abc9c;
    border-radius: 12px;
    text-align: center;
    color: #2c3e50;
    margin-top: 1rem;
}
section[data-testid="stSidebar"] {
    background-color: #0a0a12 !important;
    border-right: 2px solid #222;
    box-shadow: 4px 0 25px rgba(245, 0, 255, 0.5);
    padding: 2rem 1rem 2rem 1rem;
    font-family: 'Orbitron', sans-serif !important;
}
section[data-testid="stSidebar"] * {
    font-family: 'Orbitron', sans-serif !important;
    color: #f500ff !important;
    font-size: 16px !important;
    text-shadow: 0 0 3px #f500ff;
}
section[data-testid="stSidebar"] a {
    color: #00f7ff !important;
    text-shadow: 0 0 4px #00f7ff;
}
section[data-testid="stSidebar"] a:hover {
    color: #f500ff !important;
    text-shadow: 0 0 8px #f500ff;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="highlight">
    <h1>📊 Data Modeling & AI Predictions</h1>
    <p>Upload CSVs, train powerful models, visualize insights, and integrate live SQL editing.</p>
</div>
""", unsafe_allow_html=True)

# === TABS ===
tab1, tab2, tab3 = st.tabs(["📊 Data Visualization", "📄 CSV to Database", "🧠 Model Trainer & Predictor"])

# === TAB 1: Visualization ===
with tab1:
    st.header("Interactive Data Visualization Hub")
    uploaded_file_viz = st.file_uploader("Upload CSV for Visualization", type=["csv"], key="viz")

    if uploaded_file_viz is not None:
        df_viz = pd.read_csv(uploaded_file_viz)
        st.session_state.data = df_viz
        st.write("Preview:")
        st.dataframe(df_viz)

        num_cols = df_viz.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if st.checkbox("Show Pair Plot"):
            selected = st.multiselect("Select columns (max 5)", num_cols, default=num_cols[:3])
            if 2 <= len(selected) <= 5:
                fig = sns.pairplot(df_viz[selected])
                st.pyplot(fig)

        if st.checkbox("Show Correlation Heatmap"):
            fig, ax = plt.subplots()
            sns.heatmap(df_viz[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        if st.checkbox("Histogram"):
            column = st.selectbox("Select column", num_cols)
            fig, ax = plt.subplots()
            sns.histplot(df_viz[column], kde=True, ax=ax)
            st.pyplot(fig)

        if st.checkbox("Box Plot"):
            column = st.selectbox("Box plot column", num_cols, key="box")
            fig, ax = plt.subplots()
            sns.boxplot(x=df_viz[column], ax=ax)
            st.pyplot(fig)

        if st.checkbox("t-SNE / UMAP Clustering"):
            method = st.radio("Choose reduction method", ["t-SNE", "UMAP"])
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            df_clean = df_viz[num_cols].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_clean)
            reducer = TSNE(n_components=2, random_state=42) if method == "t-SNE" else umap.UMAP(n_components=2)
            X_reduced = reducer.fit_transform(X_scaled)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_reduced)
            fig, ax = plt.subplots()
            ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='tab10')
            ax.set_title(f"{method} Clustering")
            st.pyplot(fig)

# === TAB 2: CSV to DB ===
with tab2:
    st.header("CSV to SQLite Database")
    uploaded_file_db = st.file_uploader("Upload a CSV file", type=["csv"], key="db")
    if uploaded_file_db is not None:
        df = pd.read_csv(uploaded_file_db)
        st.session_state.data = df
        st.dataframe(df)
        conn = sqlite3.connect("uploaded_data.db")
        df.to_sql("UserData", conn, if_exists="replace", index=False)
        st.write("### Input New Data")
        input_data = {col: st.text_input(f"Enter value for {col}", key=col) for col in df.columns[:21]}
        if st.button("Submit New Row"):
            row = [input_data.get(col, None) for col in df.columns]
            conn.execute(f"INSERT INTO UserData VALUES ({','.join('?'*len(row))})", row)
            conn.commit()
            st.success("Row added.")
        updated_df = pd.read_sql("SELECT * FROM UserData", conn)
        st.dataframe(updated_df)
        st.download_button("Download CSV", updated_df.to_csv(index=False).encode("utf-8"), "updated_data.csv")
        conn.close()

# === TAB 3: Model Training ===
with tab3:
    st.header("AI Model Trainer & Predictor")
    uploaded_file_model = st.file_uploader("Upload CSV for Model Training", type=["csv"], key="model")
    model_choice = st.sidebar.selectbox("Choose a model", ("Logistic Regression", "Random Forest", "XGBoost"))
    if uploaded_file_model is not None:
        df = pd.read_csv(uploaded_file_model)
        st.session_state.data = df
        st.dataframe(df)
        target = st.selectbox("Target column", df.columns)
        X = pd.get_dummies(df.drop(columns=[target]))
        y = df[target]
        le = LabelEncoder() if y.dtype == 'object' else None
        y = le.fit_transform(y) if le else y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = LogisticRegression() if model_choice == "Logistic Regression" else RandomForestClassifier() if model_choice == "Random Forest" else XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train_scaled, y_train)
        st.success("Model trained!")
        model_file = f"model_{model_choice.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
        st.info(f"Model saved as {model_file}")
        st.subheader("Make a Prediction")
        inputs = {col: st.text_input(f"{col}") for col in X.columns}
        if st.button("Predict"):
            try:
                array = np.array([float(inputs[col]) for col in X.columns]).reshape(1, -1)
                array_scaled = scaler.transform(array)
                pred = model.predict(array_scaled)
                pred = le.inverse_transform(pred) if le else pred
                st.write(f"Prediction: {pred[0]}")
            except ValueError:
                st.error("Please provide valid numeric inputs.")

# === CHAT AI ASSISTANT ===
if st.session_state.data is not None and api_key:
    st.subheader("💬 Ask AI About Your Data")
    user_query = st.chat_input("Ask anything about your CSV dataset (e.g., 'show top 5 correlations')")
    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            try:
                sdf = SmartDataframe(st.session_state.data, config={"llm": OpenAI(api_token=api_key)})
                response = sdf.chat(user_query)
                if isinstance(response, pd.DataFrame):
                    st.dataframe(response, use_container_width=True)
                else:
                    st.write(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")

with st.sidebar.expander("📘 ML Learning Companion", expanded=False):
    st.markdown("### 🤖 Model Explanations")
    st.markdown("- **Logistic Regression**: Binary classification using sigmoid.")
    st.markdown("- **Random Forest**: Tree ensemble model.")
    st.markdown("- **XGBoost**: Optimized gradient boosting.")
    st.markdown("### 📈 Metric Tips")
    st.markdown("- Accuracy, Precision, Recall, F1.")
    st.markdown("### 💡 Tips")
    st.markdown("- Normalize, clean data, feature engineer.")
    st.markdown("### 🔗 Links")
    st.markdown("[Scikit-learn](https://scikit-learn.org/stable/)")
    st.markdown("[XGBoost](https://xgboost.readthedocs.io/en/stable/)")

# === AI CHAT BOX IN SIDEBAR (FIXED) ===
if st.session_state.data is not None and api_key:
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 💬 AI Data Chat")

    # Initialize chat history if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show last 3 messages
    for role, message in st.session_state.chat_history[-3:]:
        emoji = "🧠" if role == "ai" else "👤"
        st.sidebar.markdown(f"**{emoji} {role.capitalize()}:** {message}")

    # Chat input
    user_input = st.sidebar.text_input("Ask AI about your data", key="sidebar_chat_input")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        try:
            sdf = SmartDataframe(st.session_state.data, config={"llm": OpenAI(api_token=api_key)})
            response = sdf.chat(user_input)
            reply = response if isinstance(response, str) else response.to_string()
            st.session_state.chat_history.append(("ai", reply))
            st.sidebar.success("✅ AI Responded!")
        except Exception as e:
            st.session_state.chat_history.append(("ai", f"Error: {str(e)}"))
            st.sidebar.error(f"❌ Error: {str(e)}")

