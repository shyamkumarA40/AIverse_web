import streamlit as st
import pandas as pd
from PIL import Image


# ---- App Config ----
st.set_page_config(page_title="AIverse Home", layout="centered")

# ---- Dummy Login Credentials ----
USERNAME = "admin"
PASSWORD = "1234"

# ---- Session State for Login ----
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
# ---- Login Form ----
def login_form():
    st.markdown("## 🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid credentials")


# ---- Sidebar UI ----
def sidebar_ui():
    st.sidebar.markdown("""
        <h1 style='color: #00c3ff; font-family: "Trebuchet MS", sans-serif; font-size: 42px; margin-bottom: 10px;'>
            🌌 AIverse
        </h1>
        <hr style='border: 1px solid #444;' />
    """, unsafe_allow_html=True)

    # --- Contact Us section ---
    st.sidebar.markdown(
        """
        <br><strong>📞 Contact us</strong><br>
        <a href="https://discord.gg/jNMxQ2Qx" target="_blank"><img src="https://img.icons8.com/color/32/000000/discord--v2.png"/></a>
        <a href="https://twitter.com" target="_blank"><img src="https://img.icons8.com/color/32/000000/twitter--v1.png"/></a>
        <a href="https://www.instagram.com/ai_verse911/" target="_blank"><img src="https://img.icons8.com/color/32/000000/instagram-new--v1.png"/></a>
        <a href="https://t.me/skbytes" target="_blank"><img src="https://img.icons8.com/color/32/000000/telegram-app--v1.png"/></a>
        """,
        unsafe_allow_html=True
    )

    # Spacer
    st.sidebar.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)

# ---- Main Flow ----
if st.session_state.authenticated:
    sidebar_ui()
    st.title("🏠 Welcome to AIverse")
    st.write("Use the sidebar to explore different tools.")

    st.title('Download button')

    st.title('📥 Download CSV Files')

    @st.cache_data
    def data_read(data):
        return data.to_csv(index=False).encode('utf-8')

    # BMI CSV
    data_bmi = pd.read_csv('Dataset/bmi.csv')
    csv_bmi = data_read(data_bmi)
    st.download_button("Download BMI CSV", data=csv_bmi, file_name="bmi.csv", mime="text/csv")

    # American Bankruptcy
    data_bankruptcy = pd.read_csv('Dataset/american_bankruptcy (1).csv')
    csv_bankruptcy = data_read(data_bankruptcy)
    st.download_button("Download American Bankruptcy CSV", data=csv_bankruptcy, file_name="american_bankruptcy.csv", mime="text/csv")

    # Diabetes Prediction
    data_diabetes = pd.read_csv('Dataset/diabetes_prediction_dataset.csv')
    csv_diabetes = data_read(data_diabetes)
    st.download_button("Download Diabetes Prediction CSV", data=csv_diabetes, file_name="diabetes_prediction_dataset.csv", mime="text/csv")

    # GSPC Latest SNP
    data_gspc = pd.read_csv('Dataset/GSPC latest snp.csv')
    csv_gspc = data_read(data_gspc)
    st.download_button("Download GSPC Latest SNP CSV", data=csv_gspc, file_name="GSPC_latest_snp.csv", mime="text/csv")

    # Heart
    data_heart = pd.read_csv('Dataset/heart.csv')
    csv_heart = data_read(data_heart)
    st.download_button("Download Heart CSV", data=csv_heart, file_name="heart.csv", mime="text/csv")

    # House Price Prediction
    data_house = pd.read_csv('Dataset/House Price Prediction Dataset.csv')
    csv_house = data_read(data_house)
    st.download_button("Download House Price Prediction CSV", data=csv_house, file_name="House_Price_Prediction_Dataset.csv", mime="text/csv")

    # Loan Data
    data_loan = pd.read_csv('Dataset/loan_data.csv')
    csv_loan = data_read(data_loan)
    st.download_button("Download Loan Data CSV", data=csv_loan, file_name="loan_data.csv", mime="text/csv")

    # OOF Predictions
    data_oof = pd.read_csv('Dataset/oof_predss.csv')
    csv_oof = data_read(data_oof)
    st.download_button("Download OOF Predictions CSV", data=csv_oof, file_name="oof_predss.csv", mime="text/csv")

    # Openings
    data_openings = pd.read_csv('Dataset/openings.csv')
    csv_openings = data_read(data_openings)
    st.download_button("Download Openings CSV", data=csv_openings, file_name="openings.csv", mime="text/csv")

    # Openings FEN7
    data_openings_fen7 = pd.read_csv('Dataset/openings_fen7.csv')
    csv_openings_fen7 = data_read(data_openings_fen7)
    st.download_button("Download Openings FEN7 CSV", data=csv_openings_fen7, file_name="openings_fen7.csv", mime="text/csv")

    # Student Performance
    data_student = pd.read_csv('Dataset/Student_performance_10k.csv')
    csv_student = data_read(data_student)
    st.download_button("Download Student Performance CSV", data=csv_student, file_name="Student_performance_10k.csv", mime="text/csv")

    # Titanic Passengers
    data_titanic = pd.read_csv('Dataset/titanic-passengers.csv')
    csv_titanic = data_read(data_titanic)
    st.download_button("Download Titanic Passengers CSV", data=csv_titanic, file_name="titanic-passengers.csv", mime="text/csv")

else:
    # Hide sidebar with empty container
    st.markdown("""
        <style>
        [data-testid="stSidebar"] { display: none; }
        </style>
    """, unsafe_allow_html=True)
    login_form()
