import streamlit as st
import mysql.connector
import subprocess
import os

def connect_to_db():
    return mysql.connector.connect(
        host="localhost", user="root", password="root",
        database="Farming_Intelligence"
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

    # Store login status in session
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

    # If logged in, redirect to the Streamlit app (simulate Tkinter's subprocess.run)
    if st.session_state.logged_in:
        st.success("Redirecting to main app...")
        # You can either import and call the main app here:
        import streamlit_app
        streamlit_app.main()

        # OR use subprocess if it's a separate file
        # subprocess.run(["streamlit", "run", "streamlit_app.py"])
        # st.stop()  # prevent running anything else

if __name__ == "__main__":
    main()
