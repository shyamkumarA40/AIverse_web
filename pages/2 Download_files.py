import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

# ---- App Config ----
st.set_page_config(page_title="AIverse Home", layout="centered")

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

# ---- Dummy Login Credentials ----
USERNAME = "admin"
PASSWORD = "1234"

# ---- Session State for Login ----
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# ---- Login Form ----
def login_form():
    st.markdown("## 🔐 Login to <span style='color:#f500ff;'>AIverse</span>", unsafe_allow_html=True)
    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type="password")
    if st.button("💡 Enter"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.authenticated = True
            play_login_sound()
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

# ---- Sidebar UI ----
def sidebar_ui():
    st.sidebar.markdown("""
        <h1 style='color: #f500ff; font-family: Orbitron, sans-serif; font-size: 42px; text-shadow: 0 0 15px #f500ff;'>
            ⚡ AIverse
        </h1>
        <hr style='border: 1px solid #f500ff;' />
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
        <br><strong>📞 Contact us</strong><br>
        <a href="https://discord.gg/jNMxQ2Qx" target="_blank"><img src="https://img.icons8.com/color/32/discord--v2.png"/></a>
        <a href="https://twitter.com" target="_blank"><img src="https://img.icons8.com/color/32/twitter--v1.png"/></a>
        <a href="https://www.instagram.com/ai_verse911/" target="_blank"><img src="https://img.icons8.com/color/32/instagram-new--v1.png"/></a>
        <a href="https://t.me/skbytes" target="_blank"><img src="https://img.icons8.com/color/32/telegram-app--v1.png"/></a>
    """, unsafe_allow_html=True)
# ---- Cyberpunk Spinner ----
st.markdown("""
    <style>
    .loader {
        border: 6px solid #1c1f26;
        border-top: 6px solid #f500ff;
        border-right: 6px solid #00f7ff;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        animation: spin 1s linear infinite;
        margin: 40px auto;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)

# ---- Main Flow ----
if st.session_state.authenticated:
    sidebar_ui()
    st.markdown("<h1>👾 Welcome to <span style='color:#f500ff;'>AIverse</span></h1>", unsafe_allow_html=True)
    st.markdown("### 📥 Download Datasets")

    @st.cache_data
    def data_read(data):
        return data.to_csv(index=False).encode('utf-8')

    dataset_files = {
        "BMI CSV": "bmi.csv",
        "American Bankruptcy CSV": "american_bankruptcy (1).csv",
        "Diabetes Prediction CSV": "diabetes_prediction_dataset.csv",
        "GSPC Latest SNP CSV": "GSPC latest snp.csv",
        "Heart CSV": "heart.csv",
        "House Price Prediction CSV": "House Price Prediction Dataset.csv",
        "Loan Data CSV": "loan_data.csv",
        "OOF Predictions CSV": "oof_predss.csv",
        "Openings CSV": "openings.csv",
        "Openings FEN7 CSV": "openings_fen7.csv",
        "Student Performance CSV": "Student_performance_10k.csv",
        "Titanic Passengers CSV": "titanic-passengers.csv",
    }

    for label, filename in dataset_files.items():
        try:
            df = pd.read_csv(f"Dataset/{filename}")
            csv = data_read(df)
            st.download_button(f"⬇️ {label}", data=csv, file_name=filename, mime="text/csv")
        except FileNotFoundError:
            st.warning(f"⚠️ File not found: {filename}")

else:
    st.markdown("""
        <style>
        [data-testid="stSidebar"] { display: none; }
        </style>
    """, unsafe_allow_html=True)
    login_form()
