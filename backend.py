import pickle
import shap
import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
import json


class LoanInsightsGenerator:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def generate_shap_insights(self, model, X):
        try:
            # Extensive error checking
            if model is None:
                st.error("Model is None. Cannot generate SHAP insights.")
                return {}

            if X is None or len(X) == 0:
                st.error("Input data is empty or None. Cannot generate SHAP insights.")
                return {}

            # Extract the final estimator from the pipeline
            if hasattr(model, 'named_steps'):
                # Find the classifier step
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

            # Ensure transformed features for SHAP
            if hasattr(model, 'named_steps'):
                # Find the preprocessor step
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

            # Get feature names
            feature_names = X.columns.tolist()

            # Prepare feature importance dictionary
            feature_importance = {}

            # For binary classification, use the second class
            if len(shap_values.shape) > 2:
                shap_values = shap_values[1]

            # Calculate absolute mean SHAP values for feature importance
            mean_shap_values = np.abs(shap_values[0])

            for name, value in zip(feature_names, mean_shap_values):
                feature_importance[name] = float(value)

            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

            return dict(sorted_features)

        except Exception as e:
            st.error(f"Detailed Error in SHAP explanation: {e}")
            import traceback
            st.error(traceback.format_exc())
            return {}

    def generate_initial_insights(self, prediction, user_data, feature_importance):
        # Format property value and LTV ratio based on home ownership
        property_value_text = "N/A" if user_data.get(
            'person_home_ownership') == "RENT" else f"₹{user_data.get('property_value_inr', 'N/A'):,}"
        ltv_ratio_text = "N/A" if user_data.get(
            'person_home_ownership') == "RENT" else f"{user_data.get('ltv_ratio', 'N/A'):.2f}%"

        # Enhanced system prompt focused on borrower consultation
        system_prompt = f"""You are a senior loan consultant with 20 years of experience helping borrowers optimize their loan applications.
        Your job is to provide CONSTRUCTIVE FEEDBACK and ACTIONABLE ADVICE to the borrower (not the lender).

        You must analyze the loan application from the borrower's perspective and suggest improvements to increase approval chances.

        If the analysis suggests rejection risk, DO NOT just state this - instead provide specific, practical steps the borrower can take to improve their application.

        Even with negative indicators, maintain a supportive, solution-oriented tone. Your goal is to empower the borrower with knowledge and options.

        Always frame your feedback in terms of opportunities for improvement rather than pointing out flaws. Be empathetic and consider the borrower's goals.
        """

        # More comprehensive and borrower-focused context prompt
        context_prompt = f"""### 🏦 Your Loan Application Details
        - **Name:** {user_data.get('borrower_name', 'N/A')}
        - **CIBIL Score:** {user_data.get('cibil_score', 'N/A')}
        - **Annual Income:** ₹{user_data.get('original_income_inr', 'N/A'):,}
        - **Requested Loan Amount:** ₹{user_data.get('original_loan_amnt_inr', 'N/A'):,}
        - **Loan Purpose:** {user_data.get('loan_intent', 'N/A')}
        - **Property Value:** {property_value_text}
        - **Total Existing Debt:** ₹{user_data.get('total_debt_inr', 'N/A'):,}
        - **Loan-to-Value (LTV) Ratio:** {ltv_ratio_text}
        - **Debt-to-Income (DTI) Ratio:** {user_data.get('dti_ratio', 'N/A'):.2f}%
        - **Home Ownership Status:** {user_data.get('person_home_ownership', 'N/A')}
        - **Age:** {user_data.get('person_age', 'N/A')}
        - **Employment Length:** {user_data.get('person_emp_length', 'N/A')} years
        - **Credit History Length:** {user_data.get('cb_person_cred_hist_length', 'N/A')} years
        - **Interest Rate:** {user_data.get('loan_int_rate', 'N/A')}%
        - **Loan Grade:** {user_data.get('loan_grade', 'N/A')}
        - **Model Prediction:** {"LIKELY TO BE APPROVED" if prediction == 0 else "AT RISK OF REJECTION"}

        Based on your loan application details, I need you to:

        1. Provide a comprehensive assessment of your loan application from a borrower's perspective
        2. Identify the key strengths and potential weakness factors in your application  
        3. Suggest specific, actionable steps you can take to improve your approval chances
        4. Explain how each factor (DTI ratio, credit score, etc.) affects your application
        5. Provide realistic options based on your current financial situation

        The response MUST be structured with clear sections and bullet points for easy reading.
        Always address the borrower directly using "you" and "your".
        """

        try:
            # Using a try-except with fallback options to ensure we always get a response
            try:
                response = self.client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context_prompt}
                    ],
                    max_tokens=750,
                    temperature=0.5
                )
                return response.choices[0].message.content
            except Exception as e:
                # First fallback - try with simpler prompt
                try:
                    simplified_prompt = f"""As a loan consultant, provide advice to a borrower with:
                    - CIBIL Score: {user_data.get('cibil_score', 'N/A')}
                    - Annual Income: ₹{user_data.get('original_income_inr', 'N/A'):,}
                    - Loan Amount: ₹{user_data.get('original_loan_amnt_inr', 'N/A'):,}
                    - Purpose: {user_data.get('loan_intent', 'N/A')}
                    - Debt-to-Income: {user_data.get('dti_ratio', 'N/A'):.2f}%

                    How can they improve their application?"""

                    response = self.client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[
                            {"role": "user", "content": simplified_prompt}
                        ],
                        max_tokens=500,
                        temperature=0.5
                    )
                    return response.choices[0].message.content
                except:
                    # Final fallback - hardcoded response
                    return """
                    # Loan Application Analysis

                    Thank you for submitting your loan application. Based on the information provided, here are some general recommendations:

                    ## Key Factors to Consider

                    - **Your credit score** is one of the most important factors in loan approval
                    - **Debt-to-income ratio** significantly impacts your borrowing capacity
                    - **Loan purpose** can affect risk assessment and interest rates
                    - **Employment history** demonstrates stability to lenders

                    ## Recommendations

                    1. Consider paying down existing debt before applying
                    2. Check your credit report for errors that might be affecting your score
                    3. Maintain consistent employment history
                    4. Save for a larger down payment if possible

                    Please use the chat feature below to ask specific questions about your application.
                    """
        except Exception as e:
            return f"Error generating insights: {str(e)}"

    def chat_with_loan_assistant(self, context, user_query):
        # Borrower-focused system prompt with error handling guidance
        system_prompt = """You are a supportive loan advisor dedicated to helping borrowers navigate the loan application process. Your approach is:

1. BORROWER-FOCUSED: You represent the borrower's interests, not the lender's. Your primary goal is to help them secure approval or improve their financial situation.

2. EDUCATIONAL: Explain financial concepts in simple terms without financial jargon.

3. CONSTRUCTIVE: Even when discussing challenges in their application, always frame feedback as opportunities for improvement.

4. PRACTICAL: Provide specific, actionable advice that can be implemented, not vague suggestions.

5. EMPATHETIC: Acknowledge that loan applications can be stressful and maintain a supportive tone.

If you cannot confidently answer a specific question about loan requirements or processes, acknowledge this and suggest the borrower verify with their specific lender, as requirements vary between institutions.

DO NOT take the perspective of a lender or underwriter evaluating the application. You are the borrower's advocate and consultant."""

        # Chat context with clear emphasis on borrower consultation
        chat_context = f"""BORROWER'S LOAN APPLICATION DETAILS:
{context}

Remember: You are advising THE BORROWER (not evaluating as a lender). Your goal is to help them improve their application and understand the process better.

The borrower is asking: {user_query}

Provide helpful, supportive advice from a borrower's advocate perspective."""

        try:
            # Primary attempt with full context
            try:
                response = self.client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": chat_context}
                    ],
                    max_tokens=750,
                    temperature=0.6
                )
                return response.choices[0].message.content
            except Exception as primary_error:
                # Fallback with simplified prompt if the first attempt fails
                try:
                    simplified_prompt = f"""As a loan advisor helping a borrower: {user_query}"""

                    response = self.client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[
                            {"role": "system", "content": "You are a helpful loan advisor for borrowers."},
                            {"role": "user", "content": simplified_prompt}
                        ],
                        max_tokens=400,
                        temperature=0.6
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    # Final fallback for complete API failure
                    return f"""I apologize, but I'm currently having trouble accessing the loan advisory system. 

Here's a general response to your question about "{user_query}":

When applying for loans, it's important to maintain a good credit score, keep your debt-to-income ratio low, and have stable employment history. Consider speaking with a financial advisor for personalized advice on your specific situation.

Please try asking your question again in a moment."""
        except Exception as e:
            return f"Error generating response: {str(e)}"


def load_model(model_path):
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'pipeline_1.pkl' is in the correct directory.")
        return None