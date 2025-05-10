import streamlit as st

# ---- App Config ----
st.set_page_config(page_title="AIverse Home", layout="centered")

# ---- Dummy Login Credentials ----
USERNAME = "admin"
PASSWORD = "1234"
# ---- Cyberpunk Background and CSS ----
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');

    html, body, [class*="css"] {
        background-color: #0b0c10;
        color: #f500ff;
        font-family: 'Orbitron', sans-serif;
    }

    body::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        height: 100vh;
        width: 100vw;
        background: url("https://cdn.pixabay.com/animation/2022/10/04/17/33/17-33-35-3_512.gif") no-repeat center center fixed;
        background-size: cover;
        opacity: 0.1;
        z-index: -1;
    }

    h1, h2, h3 {
        color: #f500ff;
        text-shadow: 0 0 15px #f500ff, 0 0 25px #00f7ff;
    }

    .stTextInput > div > div > input,
    .stPasswordInput > div > div > input {
        background-color: #1c1f26;
        color: #00f7ff;
        border: 1px solid #f500ff;
        border-radius: 8px;
    }

    .stButton>button {
        background-color: #f500ff;
        color: black;
        border-radius: 10px;
        border: none;
        padding: 0.5em 1em;
        box-shadow: 0 0 15px #f500ff, 0 0 30px #00f7ff;
        transition: all 0.3s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #00f7ff;
        color: black;
        box-shadow: 0 0 15px #00f7ff, 0 0 40px #f500ff;
    }

    /* --- Cyberpunk Sidebar --- */
    [data-testid="stSidebar"] {
        background-color: #0a0a12 !important;
        border-right: 1px solid #222;
        box-shadow: 2px 0 25px rgba(245, 0, 255, 0.5);
        padding: 1rem;
    }

    [data-testid="stSidebar"] * {
        color: #f500ff !important;
        background-color: transparent !important;
    }

    [data-testid="stSidebar"] a {
        color: #00f7ff !important;
        text-shadow: 0 0 2.5px #00f7ff;
        transition: color 0.2s ease;
    }

    [data-testid="stSidebar"] a:hover {
        color: #f500ff !important;
        text-shadow: 0 0 10px #f500ff;
    }
    </style>
""", unsafe_allow_html=True)
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
