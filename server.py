from flask import Flask, request, jsonify, render_template, redirect
import json

app = Flask(__name__)

WALLET_SESSION_FILE = "wallet_session.json"
STREAMLIT_URL = "http://localhost:8501"  # Change this if deployed

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/wallet-login", methods=["POST"])
def wallet_login():
    data = request.get_json()
    wallet_address = data.get("wallet")

    if wallet_address:
        # Save wallet session
        with open(WALLET_SESSION_FILE, "w") as f:
            json.dump({"wallet": wallet_address}, f)
        return jsonify({"status": "success", "redirect": STREAMLIT_URL})
    else:
        return jsonify({"status": "failed", "error": "No wallet provided"}), 400

if __name__ == "__main__":
    app.run(port=5001, debug=True)

