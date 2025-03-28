import app as st
import pandas as pd
import pickle
import shap
import numpy as np
from groq import Groq
import json

# Hardcoded Groq API Key (IMPORTANT: Be cautious about sharing this)
GROQ_API_KEY = "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"  # Replace with your actual Groq API key

# Load the trained model pipeline
MODEL_PATH = r"C:\SatyamsFolder\projects\ML\Credit_Underwriting\pipeline_1.pkl"
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure 'Model_pipeline.pkl' is in the correct directory.")
    model = None


class LoanInsightsGenerator:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    # Modify the generate_shap_insights method
    def generate_shap_insights(self, model, X):
        try:
            # Extract the final estimator from the pipeline
            if hasattr(model, 'named_steps'):
                # Try to find the final estimator in the pipeline
                classifier_key = None
                for key, step in model.named_steps.items():
                    if hasattr(step, 'predict_proba'):
                        classifier_key = key
                        break

                if classifier_key is None:
                    st.error("No classifier found in the pipeline")
                    return {}

                final_estimator = model.named_steps[classifier_key]
            else:
                final_estimator = model

            # Ensure we're using the transformed features for SHAP
            if hasattr(model, 'named_steps'):
                # Find the preprocessor step (this might need adjustment based on your pipeline)
                preprocessor_key = None
                for key, step in model.named_steps.items():
                    if hasattr(step, 'transform'):
                        preprocessor_key = key
                        break

                if preprocessor_key:
                    X_transformed = model.named_steps[preprocessor_key].transform(X)
                else:
                    X_transformed = X
            else:
                X_transformed = X

            # Now use the final estimator with TreeExplainer
            explainer = shap.TreeExplainer(final_estimator)
            shap_values = explainer.shap_values(X_transformed)

            # Get feature names (might need adjustment)
            feature_names = X.columns.tolist()

            # Prepare feature importance dictionary
            feature_importance = {}

            # For binary classification, use the second class (usually the positive/rejected class)
            if len(shap_values.shape) > 2:
                shap_values = shap_values[1]

            # Calculate absolute mean SHAP values for feature importance
            mean_shap_values = np.abs(shap_values[0])

            for name, value in zip(feature_names, mean_shap_values):
                feature_importance[name] = value

            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

            return dict(sorted_features)

        except Exception as e:
            st.error(f"Error in SHAP explanation: {e}")
            return {}

    def generate_initial_insights(self, prediction, user_data, feature_importance):
        system_prompt = """You are a senior loan underwriter with 20 years of experience in credit risk assessment. 
        Provide a hyper-detailed, data-driven analysis of the loan application that:
        - Quantifies risk with precision
        - Offers actionable, personalized recommendations
        - Explains financial concepts clearly
        - Balances risk assessment with applicant potential"""

        context_prompt = f"""Loan Application Comprehensive Analysis:

    Prediction: {'APPROVED' if prediction == 0 else 'REJECTED'}

    DETAILED APPLICANT PROFILE:
    - Age: {user_data['person_age']} years
    - Annual Income: ${user_data['person_income']:,}
    - Employment Length: {user_data['person_emp_length']} years
    - Credit History: {user_data['cb_person_cred_hist_length']} years
    - Home Ownership: {user_data['person_home_ownership']}
    - Loan Amount: ${user_data['loan_amnt']:,}
    - Loan Purpose: {user_data['loan_intent']}
    - Loan Grade: {user_data['loan_grade']}
    - Interest Rate: {user_data['loan_int_rate']}%

    FEATURE IMPORTANCE BREAKDOWN:
    {json.dumps(feature_importance, indent=2)}

    ANALYSIS REQUIREMENTS:
    1. Provide a risk-adjusted assessment
    2. Quantify approval probability
    3. Identify specific strengths and weaknesses
    4. Offer targeted improvement strategies
    5. Compare against industry lending standards"""

        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_prompt}
                ],
                max_tokens=750,
                temperature=0.5  # Slightly lower for more precise output
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating insights: {str(e)}"

def main():
    st.title("🎯 Intelligent Loan Approval Prediction")
    st.write("Analyze your loan application with AI-powered insights")

    # Loan Application Input Section
    st.markdown("### 📋 Applicant Information")

    # Create columns for input
    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Applicant's current age")
        home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        loan_amnt = st.number_input("Requested Loan Amount ($)", min_value=0, value=10000, step=500)
        loan_intent = st.selectbox("Loan Purpose",
                                   ["MEDICAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT", "VENTURE", "PERSONAL",
                                    "EDUCATION"])
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=10,
                                                     max_value=60)

    with col2:
        person_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
        person_emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
        loan_int_rate = st.slider("Interest Rate (%)", min_value=0.0, max_value=30.0, value=5.0, step=0.1)
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])

    # Permanently set default on file as "N"
    cb_person_default_on_file = "N"

    # Prediction and Insights Button
    if st.button("Analyze Loan Application"):
        if model is not None:
            try:
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

                # Generate and display initial insights
                st.markdown("### 🔍 Detailed Loan Application Insights")

                # Display feature importance as a bar chart
                st.subheader("Feature Importance")
                feat_df = pd.DataFrame.from_dict(feature_importance, orient='index', columns=['Importance'])
                feat_df = feat_df.sort_values('Importance', ascending=False)
                st.bar_chart(feat_df)

                # Generate and display textual insights
                initial_insights = insights_generator.generate_initial_insights(
                    prediction,
                    user_data,
                    feature_importance
                )
                st.info(initial_insights)

                # Chat interface for follow-up questions
                st.markdown("### 💬 Have More Questions?")
                st.info("You can now ask follow-up questions about your loan application.")

            except Exception as e:
                st.error(f"❌ Error during assessment: {e}")
        else:
            st.error("Model is not loaded. Cannot perform prediction.")


if __name__ == "__main__":
    main()