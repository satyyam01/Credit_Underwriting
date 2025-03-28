import streamlit as st
import re
from database import UserDatabase


def validate_email(email):
    """Simple email validation"""
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None


def validate_password(password):
    """
    Password validation:
    - At least 8 characters
    - Contains at least one uppercase letter
    - Contains at least one lowercase letter
    - Contains at least one number
    """
    return (
            len(password) >= 8 and
            any(c.isupper() for c in password) and
            any(c.islower() for c in password) and
            any(c.isdigit() for c in password)
    )


def login_page():
    """Login page UI"""
    st.title("🏦 Loan Approval Prediction App - Login")

    # Initialize database
    db = UserDatabase()

    # Login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

        if login_button:
            if not username or not password:
                st.error("Please fill in all fields")
            elif db.login_user(username, password):
                # Store login state
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password")

    # Signup link
    if st.button("Create New Account"):
        st.session_state.page = 'signup'
        st.rerun()


def signup_page():
    """Signup page UI"""
    st.title("🏦 Loan Approval Prediction App - Sign Up")

    # Initialize database
    db = UserDatabase()

    # Signup form
    with st.form("signup_form"):
        new_username = st.text_input("Choose a Username")
        email = st.text_input("Email Address")
        new_password = st.text_input("Create Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        signup_button = st.form_submit_button("Sign Up")

        if signup_button:
            # Validation checks
            if not new_username or not email or not new_password or not confirm_password:
                st.error("Please fill in all fields")
            elif db.user_exists(new_username):
                st.error("Username already exists")
            elif not validate_email(email):
                st.error("Invalid email address")
            elif not validate_password(new_password):
                st.error("Password must be at least 8 characters long and contain uppercase, lowercase, and number")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                # Attempt to register user
                if db.register_user(new_username, new_password, email):
                    st.success("Account created successfully! Please log in.")
                    st.session_state.page = 'login'
                    st.rerun()
                else:
                    st.error("Registration failed. Please try again.")

    # Back to login
    if st.button("Back to Login"):
        st.session_state.page = 'login'
        st.rerun()


def homepage():
    """Main homepage with welcome and authentication options"""
    st.title("🏦 Loan Prediction Intelligence")

    # Add a cool background image or animation-like effect
    st.markdown("""
    <style>
    .homepage-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .welcome-text {
        font-size: 24px;
        color: #333;
        margin-bottom: 20px;
    }
    </style>
    <div class="homepage-container">
        <div class="welcome-text">
            Welcome to Loan Prediction Intelligence 🚀
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Description
    st.write("""
    ### Intelligent Loan Application Analysis

    Our AI-powered platform provides:
    - Comprehensive loan application assessment
    - Instant risk analysis
    - Personalized financial insights
    - Advanced machine learning predictions
    """)

    # Authentication buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login", key="login_btn", use_container_width=True):
            st.session_state.page = 'login'
            st.rerun()

    with col2:
        if st.button("Sign Up", key="signup_btn", use_container_width=True):
            st.session_state.page = 'signup'
            st.rerun()


def main():
    """Main application flow"""
    # Initialize session state variables if not exist
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Routing based on session state
    if st.session_state.logged_in:
        # Import the main app here to avoid circular imports
        from app import main as app_main
        app_main()
    else:
        # Routing for authentication pages
        if st.session_state.page == 'home':
            homepage()
        elif st.session_state.page == 'login':
            login_page()
        elif st.session_state.page == 'signup':
            signup_page()


if __name__ == "__main__":
    main()