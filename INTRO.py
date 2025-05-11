import streamlit as st
import time
import json
import os
import webbrowser
from PIL import Image
import time
# Constants
USERNAME = "admin"
PASSWORD = "admin"
WALLET_SESSION_FILE = "wallet_session.json"

# Page config
st.set_page_config(page_title="AIverse - Login")
# ---- Cyberpunk Background and CSS ----
st.markdown("""
<style>
/* === FONT IMPORT === */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');

/* === MAIN PAGE LIGHT THEME === */
html, body, .stApp {
    background: linear-gradient(to bottom right, #0f051d, #190933, #2c0f4a);
    background-attachment: fixed;
    background-size: cover;
    color: #f0f0f0 !important;
    font-family: 'Segoe UI', sans-serif !important;
}


/* === CYBERPUNK GHOST BACKGROUND === */
body::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    height: 100vh;
    width: 100vw;
    background: url("https://cdn.pixabay.com/animation/2022/10/04/17/33/17-33-35-3_512.gif") no-repeat center center fixed;
    background-size: cover;
    opacity: 0.05;
    z-index: -1;
}

/* === HEADERS STYLED === */
h1, h2, h3 {
    color: #2c3e50;
}

/* === GLOWING HEADER EFFECT === */
.glow-text {
    font-family: 'Orbitron', sans-serif;
    color: #00f7ff;
    text-shadow:
        0 0 5px #00f7ff,
        0 0 10px #00f7ff,
        0 0 20px #f500ff,
        0 0 30px #f500ff;
    animation: glowPulse 2s ease-in-out infinite alternate;
}

@keyframes glowPulse {
    from {
        text-shadow:
            0 0 5px #00f7ff,
            0 0 10px #00f7ff,
            0 0 20px #f500ff;
    }
    to {
        text-shadow:
            0 0 10px #00f7ff,
            0 0 20px #00f7ff,
            0 0 40px #f500ff;
    }
}

/* === HEADER BANNER === */
html, body, .stApp {
    background: linear-gradient(to bottom right, #0f051d, #190933, #2c0f4a);
    background-attachment: fixed;
    background-size: cover;
    color: #f0f0f0 !important;
    font-family: 'Segoe UI', sans-serif !important;
}

            
.highlight h1 {
    font-size: 2.6rem;
    margin: 0;
}

/* === DATA INFO CARD === */
.data-box {
    background-color: #ffffff;
    padding: 1.5rem;
    border: 2px dashed #1abc9c;
    border-radius: 12px;
    text-align: center;
    color: #2c3e50;
    margin-top: 1rem;
}

/* === INPUT FIELDS === */
.stTextInput > div > div > input,
.stPasswordInput > div > div > input {
    background-color: #1c1f26;
    color: #00f7ff;
    border: 1px solid #f500ff;
    border-radius: 8px;
}

/* === BUTTON STYLING === */
.stButton>button {
    background-color: #f500ff;
    color: white !important;  /* Fix visibility */
    border-radius: 10px;
    border: none;
    padding: 0.5em 1em;
    box-shadow: 0 0 15px #f500ff, 0 0 30px #00f7ff;
    transition: all 0.3s ease-in-out;
}
.stButton>button:hover {
    background-color: #00f7ff;
    color: black !important;
    box-shadow: 0 0 15px #00f7ff, 0 0 40px #f500ff;
}

/* === SIDEBAR CYBERPUNK OVERRIDES === */
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
<style>
/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0f0f0f;
    background-image: radial-gradient(circle at 20% 20%, rgba(0, 255, 255, 0.1), transparent),
                      radial-gradient(circle at 80% 80%, rgba(255, 0, 255, 0.1), transparent);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
    color: white;
}

/* Sidebar Header Styling */
[data-testid="stSidebar"] h2 {
    font-size: 24px;
    font-weight: bold;
    color: #0ff;
    text-shadow: 0 0 5px #0ff, 0 0 10px #0ff;
}

/* Sidebar Buttons */
section[data-testid="stSidebar"] button {
    background-color: #111111;
    border: 1px solid #0ff;
    color: white !important; /* Force text color to white */
    font-weight: bold;
    box-shadow: 0 0 10px #0ff, 0 0 20px #0ff;
    transition: all 0.3s ease-in-out;
    padding: 0.5rem 1rem;
    border-radius: 8px;
}

/* Hover effect */
section[data-testid="stSidebar"] button:hover {
    background-color: #222222;
    box-shadow: 0 0 15px #0ff, 0 0 30px #0ff;
    color: #ffffff !important;
}
</style>

""", unsafe_allow_html=True)

# Wallet session functions
def get_wallet_session():
    if os.path.exists(WALLET_SESSION_FILE):
        with open(WALLET_SESSION_FILE, "r") as f:
            data = json.load(f)
            return data.get("wallet")
    return None

def disconnect_wallet():
    if os.path.exists(WALLET_SESSION_FILE):
        os.remove(WALLET_SESSION_FILE)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Disconnected successfully.")
    st.rerun()

def wallet_login():
    st.markdown("## 🦊 Login with MetaMask")
    if st.button("Login with MetaMask"):
        webbrowser.open_new_tab("http://localhost:5001/")
        st.info("MetaMask login page opened in a new tab. Waiting for wallet connection...")

        with st.spinner("Waiting for wallet connection..."):
            for _ in range(30):  # Wait 15 seconds max
                wallet_address = get_wallet_session()
                if wallet_address:
                    st.session_state.authenticated = True
                    st.session_state.wallet = wallet_address
                    st.rerun()
                time.sleep(0.5)

        st.warning("Wallet not connected. Please try again.")
import streamlit as st
import base64
import webbrowser
import time

# Function to inject background image CSS
def add_login_background(image_file):
    with open(image_file, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode()

    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{base64_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


def login_form():
    add_login_background("images/background40.jpg")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧑 Username Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", key="login_form_button"):
            if username == USERNAME and password == PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials")

    with col2:
       
        wallet_login()

    if st.button("Back to Home", key="back_to_home_button"):
        st.session_state.show_login_form = False
        st.rerun()


def show_authenticated_content():
    st.title("🎉 Welcome to AIverse, Explorer!")
    from PIL import Image


    # Inject Custom CSS
    st.markdown("""
    <style>
    body {
        background-color: #f1f2f6;
    }
    html, body, .stApp {
        background: linear-gradient(to bottom right, #0f051d, #190933, #2c0f4a);
        background-attachment: fixed;
        background-size: cover;
        color: #f0f0f0 !important;
        font-family: 'Segoe UI', sans-serif !important;
    }

    .hero h1 {
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }
    .hero p {
        font-size: 1.2rem;
    }
html, body, .stApp {
    background: linear-gradient(to bottom right, #0f051d, #190933, #2c0f4a);
    background-attachment: fixed;
    background-size: cover;
    color: #f0f0f0 !important;
    font-family: 'Segoe UI', sans-serif !important;
}
    .section h2 {
        color: #2f3542;
        margin-bottom: 1rem;
    }
    .section ul {
        line-height: 1.8;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #7f8fa6;
        font-size: 0.9rem;
    }
    </style>

    """, unsafe_allow_html=True)

    # Hero Banner
    st.markdown("""
    <div class="hero">
        <h1>🚀 AI Fusion Hub</h1>
        <p>Unifying Data Intelligence, Voice Emotion AI, and Facial Mood Detection into a sleek, interactive platform.</p>
    </div>
    """, unsafe_allow_html=True)

    # Section 1: DATA MODELS
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 📊 Data Modeling & Predictions")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("assets/data_model_image.jpg", caption="Train and Visualize", use_container_width=True)
    with col2:
        st.markdown("""
        **Core Features:**
        - Upload CSV to preview and clean data
        - Train with Logistic Regression, Random Forest, XGBoost
        - Auto-update model with new uploaded data
        - Live SQL Editing to add/modify data
        - Visualize data via bar plots, heatmaps, line and scatter charts
        """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: VOICE MODEL
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 🎙️ Voice Emotion & Gender Analysis")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **How it Works:**
        - Upload `.wav` audio file
        - Detect speaker’s **gender** and **emotional tone**
        - View waveform and pitch curves
        - Read auto-generated speech **transcript**
        - Great for voice UX testing and emotion-aware apps
        """)
    with col2:
        st.image("assets/voice_model_image.jpg", caption="Understand Emotions Through Sound", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Section 3: IMG MOOD AI
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("## 😊 Image-Based Emotion Detection")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("assets/image_mood_ai.jpg", caption="Facial Mood Recognition", use_container_width=True)
    with col2:
        st.markdown("""
        **Use Case:**
        - Upload a clear face image
        - Detect facial emotion (Happy, Sad, Angry, Neutral, etc.)
        - AI responds with a relevant reaction (visual/gif/text)
        - Ideal for user mood detection, mental health, smart UIs
        """)
    st.markdown('</div>', unsafe_allow_html=True)
def sidebar_ui():
    st.sidebar.title("🔐 AIverse Authentication")
    st.sidebar.markdown("Welcome to **AIverse**! Please log in using one of the methods below.")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
    

        if st.sidebar.button("Logout"):
            disconnect_wallet()

        if st.session_state.get("wallet"):
            st.sidebar.markdown(f"""
            <hr />
            <small style="color: #00e1ff;"><strong>🦊 Wallet:</strong><br>{st.session_state.wallet}</small>
            """, unsafe_allow_html=True)
    else:
        if st.sidebar.button("Login"):
            st.session_state.show_login_form = True
            st.rerun()

# Check wallet session at start
wallet_address = get_wallet_session()
if wallet_address and not st.session_state.get("authenticated"):
    st.session_state.authenticated = True
    st.session_state.wallet = wallet_address

# UI Rendering
sidebar_ui()

if st.session_state.get("authenticated"):
    show_authenticated_content()

elif st.session_state.get("show_login_form"):
    login_form()
else:
    import base64
    from PIL import Image
    from io import BytesIO
    def gif_to_base64(gif_path):
        with open(gif_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:image/gif;base64,{encoded}"

    # Convert two different GIFs to base64
    gif1_base64 = gif_to_base64("images/com-crop-unscreen.gif")  # Replace with your actual path
    gif2_base64 = gif_to_base64("images/com-crop-unscreen-ezgif.com-rotate.gif")  # Replace with your actual path

    # Display HTML + CSS with embedded GIFs
    st.markdown(f"""
    <style>
    .hero-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1px;
        margin: 60px 0;
        flex-wrap: wrap;
    }}

    .hero-text {{
        text-align: center;
        color: white;
        font-family: 'Segoe UI', sans-serif;
        text-shadow: 1px 1px 5px rgba(0,0,0,0.4);
    }}

    .hero-text h1 {{
        font-size: 64px;
        margin: 0;
        color: #00f0ff;
    }}

    .hero-text h2 {{
        font-size: 50px;
        margin: 0;
        color: #ffffff;
    }}
    
    .hero-text h3 {{
        font-size: 42px;
        margin-top: 9px;
        color: #b2ebf2;
    }}

    .hero-image {{
        max-width: 90px;
        height: 250px;
    }}
  

    </style>
    

    <div class="hero-container">
        <img src="{gif1_base64}" class="hero-image" />
        <div class="hero-text">
            <h1>Step Into</h1>
            <h2>The AIverse - A Universe</h2>
            <h3>Where AI Reads You</h3>
        </div>
        <img src="{gif2_base64}" class="hero-image" />
    </div>
    """, unsafe_allow_html=True)



    st.markdown("--------------------")
    st.markdown("-------------")
    from base64 import b64encode
    # --- Combined Global and Custom CSS Styling ---
    st.markdown("""
        <style>
            html, body, [class*="css"]  {
                font-family: 'Segoe UI', sans-serif;
                font-size: 16px;
                background-color: #F5F5F5;
            }










            .stFileUploader, .stAudio, .stMarkdown {
                font-size: 18px;
            }
            .stButton, .stCheckbox {
                font-size: 14px;
            }
            .custom-box {
                font-size: 20px;
                padding: 1px;
                background-color: #212121;
                border-radius: 5px;
            }
            .card-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                justify-content: center;
                margin-top: 30px;
            }
            .card {
                background-color: #ffffff;
                border-radius: 15px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                padding: 20px;
                width: 300px;
                transition: transform 0.2s;
            }
            .card:hover {
                transform: scale(1.07);
            }
            .card-title {
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #333;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .card-content {
                font-size: 16px;
                color: #666;
            }
            .icon {
                width: 24px;
                height: 24px;
            }
            @media screen and (max-width: 600px) {
                .card-container {
                    flex-direction: column;
                    align-items: center;
                }
            }
        </style>
    """, unsafe_allow_html=True)

    # --- Dashboard Cards ---
    st.markdown("""
        <div class="card-container">
            <div class="card">
                <div class="card-title">
                    <img src="https://img.icons8.com/ios-filled/50/4A90E2/dashboard.png" class="icon"/>
                    Metamask Wallet
                </div>
                <div class="card-content">
                    Login via Metamask and experiance Web3.
                </div>
            </div>
            <div class="card">
                <div class="card-title">
                    <img src="https://img.icons8.com/ios-filled/50/4A90E2/data-configuration.png" class="icon"/>
                    SQL DATABASE IN CSV
                </div>
                <div class="card-content">
                    Update Data ,  Download Updated csv and Train Models.
                </div>
            </div>
            <div class="card">
                <div class="card-title">
                    <img src="https://img.icons8.com/ios-filled/50/4A90E2/statistics.png" class="icon"/>
                    Voice/Image Emotion Prediction
                </div>
                <div class="card-content">
                    Explore the Emotion AI.
                </div>
            </div>
            <div class="card">
                <div class="card-title">
                    <img src="https://img.icons8.com/ios-filled/50/4A90E2/support.png" class="icon"/>
                    AI Chat Terminal
                </div>
                <div class="card-content">
                    Uploaded CSV file is connected to Chat terminal, easy to edit modify etc.. via text.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("-------------")
    st.markdown("-------------")

    # --- Section 1: Voice Recognition ---
    col1, col2 = st.columns([1, 1])
    with col1:
 
        st.markdown('<div class="custom-header">🎤 AI Voice Recognition</div>', unsafe_allow_html=True)
        st.markdown("""
        - Transcribe speech to text in real-time  
        - Detect tone and emotion in your voice  
        - Support multilingual input  
        """)
        st.markdown("""
        Built using powerful neural networks trained on diverse datasets.
        """)

        # ✅ Replace HTML button with a real Streamlit button
        if st.button("🎯 Login to access", key="voice_login_button"):
            st.session_state.page = "login"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.image("pages/alexa-amazon.gif", caption="Emotion Prediction via Voice", use_container_width=True)

    st.markdown("-------------")
    st.markdown("-------------")

    # --- Section 2: Emotion from Image ---
    col3, col4 = st.columns([1, 1])
    with col3:
        st.image("pages/emotion.gif", caption="Emotion Prediction via Image", use_container_width=True)

    with col4:
        st.markdown('<div class="custom-header">🖼️ Emotion Prediction from Image</div>', unsafe_allow_html=True)
        st.markdown("""
        Just upload an image or a GIF showing a person's face, and our AI model will:

        - Analyze facial expressions  
        - Predict the underlying emotion *(happy, sad, angry, surprised, etc.)*  
        - Work with both static images and animated GIFs  
        """)
        st.markdown("Explore the emotional dimension of AI!")
        if st.button("🎯 Login to access", key="emotion_login_button"):
            st.session_state.page = "login"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("-------------")

    



