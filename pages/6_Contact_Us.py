import streamlit as st

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
    st.markdown("""
        <br><strong>📞 Contact us</strong><br>
        <a href="https://discord.gg/jNMxQ2Qx" target="_blank"><img src="https://img.icons8.com/color/32/000000/discord--v2.png"/></a>
        <a href="https://twitter.com" target="_blank"><img src="https://img.icons8.com/color/32/000000/twitter--v1.png"/></a>
        <a href="https://instagram.com" target="_blank"><img src="https://img.icons8.com/color/32/000000/instagram-new--v1.png"/></a>
        <a href="https://t.me/skbytes" target="_blank"><img src="https://img.icons8.com/color/32/000000/telegram-app--v1.png"/></a>
    """, unsafe_allow_html=True)

else:
    # Hide sidebar with empty container
    st.markdown("""
        <style>
        [data-testid="stSidebar"] { display: none; }
        </style>
    """, unsafe_allow_html=True)
    login_form()
