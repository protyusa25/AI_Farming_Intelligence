# login_app.py
import streamlit as st
import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

def connect_to_db():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

def signup(username, password):
    if not username or not password:
        st.warning("Please enter both username and password.")
        return
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        conn.commit()
        conn.close()
        st.success("Signup successful! Please login.")
    except mysql.connector.IntegrityError:
        st.error("Username already exists.")
    except Exception as e:
        st.error(f"Error: {e}")

def login(username, password):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()
        conn.close()
        return user is not None
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

def main():
    st.set_page_config(page_title="Login - AI Helper for Farmers")
    st.title("🌿 AI Helper for Farmers - Authentication")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔑 Login")
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if login(login_user, login_pass):
                st.success(f"Welcome, {login_user}!")
                st.session_state.logged_in = True
                st.session_state.username = login_user
            else:
                st.error("Invalid credentials.")

    with col2:
        st.subheader("📝 Signup")
        signup_user = st.text_input("New Username", key="signup_user")
        signup_pass = st.text_input("New Password", type="password", key="signup_pass")
        if st.button("Signup"):
            signup(signup_user, signup_pass)

    if st.session_state.logged_in:
        st.success("Redirecting to main app...")
        import streamlit_app  # Make sure this exists!
        streamlit_app.main()

if __name__ == "__main__":
    main()
