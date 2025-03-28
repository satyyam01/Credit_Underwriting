import streamlit as st
import pandas as pd
import json
import requests
from backend import load_model, LoanInsightsGenerator
from chatbot import initialize_chat_session, display_chat_history, handle_chat_interaction

# Hardcoded Groq API Key (IMPORTANT: Be cautious about sharing this)
GROQ_API_KEY = "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"  # Replace with your actual Groq API key

# Load the trained model pipeline
MODEL_PATH = r"pipeline_1.pkl"  # Update this path as needed
model = load_model(MODEL_PATH)


def get_exchange_rate():
    """
    Fetch current INR to USD exchange rate.
    Fallback to a recent approximate rate if API fails.
    """
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/INR")
        return response.json()['rates']['USD']
    except:
        # Fallback exchange rate (approximate)
        return 0.012  # As of 2024, 1 INR ≈ 0.012 USD


def calculate_loan_grade(cibil_score):
    """
    Calculate loan grade based on CIBIL score
    CIBIL Score ranges:
    - 300-579: Poor (G)
    - 580-669: Fair (F)
    - 670-739: Good (D)
    - 740-799: Very Good (B)
    - 800-900: Excellent (A)
    """
    if cibil_score < 580:
        return 'G'
    elif cibil_score < 670:
        return 'F'
    elif cibil_score < 740:
        return 'D'
    elif cibil_score < 800:
        return 'B'
    else:
        return 'A'


def logout():
    """Logout functionality"""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun()


def main():
    # Exchange rate (fetch or use fallback)
    exchange_rate = get_exchange_rate()

    # Add sidebar with user information
    st.sidebar.title(f"👤 Welcome, {st.session_state.username}")
    st.sidebar.markdown("---")

    # Add logout button in sidebar
    if st.sidebar.button("🚪 Logout"):
        logout()

    st.title("🎯 Intelligent Loan Approval Prediction")
    st.write("Analyze your loan application with AI-powered insights")

    # Initialize session state for tracking analysis state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False

    # Loan Application Input Section
    st.markdown("### 📋 Applicant Information")

    # Create columns for input
    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Applicant's current age")
        home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])

        # Loan Amount in INR, convert to USD
        loan_amnt_inr = st.number_input("Requested Loan Amount (₹)", min_value=0, value=10_00_000, step=50_000)
        loan_amnt = round(loan_amnt_inr * exchange_rate, 2)

        loan_intent = st.selectbox("Loan Purpose",
                                   ["MEDICAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT", "VENTURE", "PERSONAL",
                                    "EDUCATION"])
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=10,
                                                     max_value=60)

    with col2:
        # Annual Income in INR, convert to USD
        person_income_inr = st.number_input("Annual Income (₹)", min_value=0, value=10_00_000, step=50_000)
        person_income = round(person_income_inr * exchange_rate, 2)

        person_emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
        loan_int_rate = st.slider("Interest Rate (%)", min_value=0.0, max_value=30.0, value=5.0, step=0.1)

        # CIBIL Score input
        cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700,
                                      help="Credit score between 300-900")

        # Dynamically calculate loan grade based on CIBIL score
        loan_grade = calculate_loan_grade(cibil_score)
        st.info(f"Calculated Loan Grade: {loan_grade}")

    # Permanently set default on file as "N"
    cb_person_default_on_file = "N"

    # Display current exchange rate
    st.sidebar.markdown(f"💱 Current Exchange Rate: 1 INR = {exchange_rate:.4f} USD")

    # Prediction and Insights Button
    if st.button("Analyze Loan Application") or st.session_state.analysis_done:
        if model is not None:
            try:
                # If analysis is not already done, perform the analysis
                if not st.session_state.analysis_done:
                    # Prepare user input
                    user_input = pd.DataFrame([{
                        'person_age': person_age,
                        'person_income': person_income,
                        'person_home_ownership': home_ownership,
                        'person_emp_length': person_emp_length,
                        'loan_intent': loan_intent,
                        'loan_grade': loan_grade,
                        'loan_amnt': loan_amnt,
                        'loan_int_rate': loan_int_rate,
                        'cb_person_default_on_file': cb_person_default_on_file,
                        'cb_person_cred_hist_length': cb_person_cred_hist_length,
                    }])

                    # Make prediction
                    prediction = model.predict(user_input)[0]

                    # Display prediction results
                    if prediction == 0:
                        st.balloons()
                        st.success(
                            "✅ Loan is likely to get approved: The application shows good indicators for loan approval.")
                    else:
                        st.warning(
                            "⚠️ Application likely to get rejected: The application shows elevated chance of rejection.")

                    # Initialize Insights Generator
                    insights_generator = LoanInsightsGenerator(GROQ_API_KEY)

                    # Generate SHAP Feature Importance
                    feature_importance = insights_generator.generate_shap_insights(model, user_input)

                    # Prepare user data dict for insights context
                    user_data = user_input.iloc[0].to_dict()
                    user_data['original_income_inr'] = person_income_inr
                    user_data['original_loan_amnt_inr'] = loan_amnt_inr
                    user_data['cibil_score'] = cibil_score

                    # Store results in session state
                    st.session_state.prediction = prediction
                    st.session_state.feature_importance = feature_importance
                    st.session_state.user_data = user_data

                    # Generate and display initial insights
                    initial_insights = insights_generator.generate_initial_insights(
                        prediction,
                        user_data,
                        feature_importance
                    )
                    st.session_state.initial_insights = initial_insights
                    st.session_state.analysis_done = True

                # If analysis is already done, use stored results
                prediction = st.session_state.prediction
                feature_importance = st.session_state.feature_importance
                user_data = st.session_state.user_data
                initial_insights = st.session_state.initial_insights

                # Display stored results
                st.markdown("### 🔍 Detailed Loan Application Insights")

                # Display feature importance as a bar chart
                st.subheader("Feature Importance")
                feat_df = pd.DataFrame.from_dict(feature_importance, orient='index', columns=['Importance'])
                feat_df = feat_df.sort_values('Importance', ascending=False)
                st.bar_chart(feat_df)

                # Display initial insights
                st.info(initial_insights)

                # Chat interface for follow-up questions
                st.markdown("### 💬 Loan Application Chat")

                # Initialize chat session
                initialize_chat_session()

                # Display chat history
                display_chat_history()

                # Generate context for chat
                context = f"""Loan Application Details:
Prediction: {'APPROVED' if prediction == 0 else 'REJECTED'}
{json.dumps(user_data, indent=2)}
Feature Importance: {json.dumps(feature_importance, indent=2)}
Initial Insights: {initial_insights}"""

                # Handle chat interaction
                handle_chat_interaction(context, GROQ_API_KEY)

            except Exception as e:
                st.error(f"❌ Error during assessment: {e}")
                import traceback
                st.error(traceback.format_exc())
        else:
            st.error("Model is not loaded. Cannot perform prediction.")


if __name__ == "__main__":
    main()