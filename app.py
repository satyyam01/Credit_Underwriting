import streamlit as st
import pandas as pd
import json
import requests
from backend import load_model, LoanInsightsGenerator
from chatbot import initialize_chat_session, display_chat_history, handle_chat_interaction
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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


def calculate_ltv_ratio(loan_amount, property_value, home_ownership):
    """
    Calculate Loan-to-Value ratio
    Returns 0 if home ownership is RENT
    """
    if home_ownership == "RENT":
        return 0

    if property_value <= 0:
        return 0
    return (loan_amount / property_value) * 100


def calculate_dti_ratio(total_debt, annual_income):
    """
    Calculate Debt-to-Income ratio
    """
    if annual_income <= 0:
        return 0
    # DTI is typically calculated as (monthly debt / monthly income) * 100
    # Here we're simplifying by using annual values
    return (total_debt / annual_income) * 100


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

    st.title("🎯 Intelligent Loan Application Assistant")
    st.write("Get personalized advice to improve your loan approval chances")

    # Initialize session state for tracking analysis state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False

    # Loan Application Input Section
    st.markdown("### 📋 Your Information")

    # Create columns for input
    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Your current age")
        home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])

        borrower_name = st.text_input("Your Name", help="Enter your full name")
        # Loan Amount in INR, convert to USD

        loan_amnt_inr = st.number_input("Requested Loan Amount (₹)", min_value=0, value=10_00_000, step=50_000)
        loan_amnt = round(loan_amnt_inr * exchange_rate, 2)

        loan_intent = st.selectbox("Loan Purpose",
                                   ["MEDICAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT", "VENTURE", "PERSONAL",
                                    "EDUCATION"])
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=10,
                                                     max_value=60)

        # Property value input - disabled if RENT is selected
        property_value_disabled = home_ownership == "RENT"
        property_value_help = "Not applicable for RENT status" if property_value_disabled else "Current market value of property"
        property_value_inr = st.number_input(
            "Property Value (₹)",
            min_value=0,
            value=0 if property_value_disabled else 50_00_000,
            step=1_00_000,
            disabled=property_value_disabled,
            help=property_value_help
        )
        if property_value_disabled:
            property_value_inr = 0
        property_value = round(property_value_inr * exchange_rate, 2)

    with col2:
        # Annual Income in INR, convert to USD
        person_income_inr = st.number_input("Annual Income (₹)", min_value=0, value=10_00_000, step=50_000)
        person_income = round(person_income_inr * exchange_rate, 2)

        person_emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
        loan_int_rate = st.slider("Interest Rate (%)", min_value=0.0, max_value=30.0, value=5.0, step=0.1)

        # CIBIL Score input
        cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700,
                                      help="Credit score between 300-900")

        # New input for total existing debt
        total_debt_inr = st.number_input("Total Existing Debt (₹)", min_value=0, value=5_00_000, step=50_000,
                                         help="Sum of all current outstanding debts")
        total_debt = round(total_debt_inr * exchange_rate, 2)

        # Dynamically calculate loan grade based on CIBIL score
        loan_grade = calculate_loan_grade(cibil_score)
        st.info(f"Calculated Loan Grade: {loan_grade}")

    # Permanently set default on file as "N"
    cb_person_default_on_file = "N"

    # Calculate LTV ratio considering home ownership
    ltv_ratio = calculate_ltv_ratio(loan_amnt_inr, property_value_inr, home_ownership)

    # Calculate DTI ratio
    dti_ratio = calculate_dti_ratio(total_debt_inr, person_income_inr)

    # Display financial ratios in sidebar
    st.sidebar.markdown("### 📊 Your Financial Ratios")
    if home_ownership == "RENT":
        st.sidebar.markdown("LTV Ratio: Not Applicable (Rental)")
    else:
        st.sidebar.markdown(f"LTV Ratio: {ltv_ratio:.2f}%")
    st.sidebar.markdown(f"DTI Ratio: {dti_ratio:.2f}%")

    # Display current exchange rate
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"💱 Current Exchange Rate: 1 INR = {exchange_rate:.4f} USD")

    # Prediction and Insights Button
    if st.button("Analyze My Application") or st.session_state.analysis_done:
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
                        'borrower_name': borrower_name,
                    }])

                    # Make prediction
                    prediction = model.predict(user_input)[0]

                    # Display prediction results with borrower-friendly language
                    if prediction == 0:
                        st.balloons()
                        st.success(
                            "✅ Good News! Your application shows positive indicators for approval.")
                    else:
                        st.warning(
                            "⚠️ Your application may need improvements to increase approval chances.")

                    # Initialize Insights Generator
                    insights_generator = LoanInsightsGenerator(GROQ_API_KEY)

                    # Generate SHAP Feature Importance
                    feature_importance = insights_generator.generate_shap_insights(model, user_input)

                    # Prepare user data dict for insights context
                    user_data = user_input.iloc[0].to_dict()
                    user_data['original_income_inr'] = person_income_inr
                    user_data['original_loan_amnt_inr'] = loan_amnt_inr
                    user_data['cibil_score'] = cibil_score
                    # Add new property value and debt fields
                    user_data['property_value_inr'] = 0 if home_ownership == "RENT" else property_value_inr
                    user_data['total_debt_inr'] = total_debt_inr
                    user_data['ltv_ratio'] = ltv_ratio
                    user_data['dti_ratio'] = dti_ratio

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
                st.markdown("### 🔍 Your Personalized Loan Application Insights")

                # Display feature importance as a bar chart with explanatory text
                st.subheader("Factors Affecting Your Application")
                st.write("These factors have the most impact on your loan approval chances:")
                feat_df = pd.DataFrame.from_dict(feature_importance, orient='index', columns=['Importance'])
                feat_df = feat_df.sort_values('Importance', ascending=False)
                st.bar_chart(feat_df)

                # Display initial insights with better formatting
                st.markdown("### 📝 Personalized Recommendations")
                st.markdown(initial_insights)

                # Chat interface for follow-up questions
                st.markdown("### 💬 Ask Questions About Your Application")
                st.write("Have questions about your loan application? Ask our loan advisor for personalized guidance.")

                # Initialize chat session
                initialize_chat_session()

                # Display chat history
                display_chat_history()

                # Generate context for chat
                context = f"""Loan Application Details:
Prediction: {'LIKELY TO BE APPROVED' if prediction == 0 else 'AT RISK OF REJECTION'}
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