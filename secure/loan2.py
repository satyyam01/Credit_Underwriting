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
MODEL_PATH = r"/pipeline_1.pkl"
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure 'Model_pipeline.pkl' is in the correct directory.")
    model = None


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
        system_prompt = """You are a senior loan underwriter with 20 years of experience in credit risk assessment.
        Your task is to conduct a highly structured, data-driven risk assessment of loan applications.

        - Assign precise risk scores from 1 to 10 (1 = Low Risk, 10 = High Risk)
        - Justify each score with clear financial reasoning
        - Compare applicant’s financial standing to industry averages
        - Suggest *exact* numerical improvements (e.g., increase income by $X, reduce DTI by Y%)
        - Explain insights using easy-to-understand financial concepts"""

        context_prompt = f"""### 🏦 Loan Application Risk Assessment

    **🟢 Prediction: {'✅ APPROVED' if prediction == 0 else '❌ REJECTED'}**  

    ### 📌 **Applicant Financial Profile**
    - **Age:** {user_data.get('person_age', 'N/A')} years  
    - **Annual Income:** ${user_data.get('person_income', 'N/A'):,}  
    - **Employment Length:** {user_data.get('person_emp_length', 'N/A')} years  
    - **Credit History:** {user_data.get('cb_person_cred_hist_length', 'N/A')} years  
    - **Home Ownership:** {user_data.get('person_home_ownership', 'N/A')}  
    - **Loan Amount:** ${user_data.get('loan_amnt', 'N/A'):,}  
    - **Loan Purpose:** {user_data.get('loan_intent', 'N/A')}  
    - **Loan Grade:** {user_data.get('loan_grade', 'N/A')}  
    - **Interest Rate:** {user_data.get('loan_int_rate', 'N/A')}%  

    ### 🔍 **Feature Importance Breakdown**
    {json.dumps(feature_importance, indent=2)}

    ### 🏦 **Financial Risk Analysis**
    #### **1️⃣ Income & Debt-to-Income (DTI) Ratio**
    - **Applicant DTI Ratio:** _(calculated if available)_  
    - **Industry Benchmark:** _X%_  
    - **Risk Score (1-10):** _X/10_  
    - **Reasoning:** _Explain why their income supports/does not support the loan._  
    - **Recommendation:** _Increase income by $X or reduce outstanding debt by $Y._

    #### **2️⃣ Creditworthiness**
    - **Credit History Length:** {user_data.get('cb_person_cred_hist_length', 'N/A')} years  
    - **Industry Average:** _X years_  
    - **Risk Score (1-10):** _X/10_  
    - **Reasoning:** _Explain if credit history is sufficient._  
    - **Recommendation:** _Improve by maintaining credit utilization under Y%._

    #### **3️⃣ Loan Affordability**
    - **Requested Loan Amount:** ${user_data.get('loan_amnt', 'N/A'):,}  
    - **Affordability Index (based on income & DTI):** _X%_  
    - **Risk Score (1-10):** _X/10_  
    - **Reasoning:** _Assess if the requested amount is reasonable._  
    - **Recommendation:** _Consider reducing loan amount to ${user_data.get('loan_amnt', 'N/A') * 0.8:,} for better approval odds._

    ---

    ### 🎯 **Final Decision Justification & Action Plan**
    - **Approval Probability:** _X% based on risk factors._  
    - **Key Strengths:** _(Highlight 2-3 applicant advantages)_  
    - **Key Weaknesses:** _(Highlight 2-3 major risk factors)_  
    - **Action Plan:** _Provide 2-3 specific, numeric improvement steps._  
    """

        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_prompt}
                ],
                max_tokens=750,
                temperature=0.5  # Balanced for precision
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating insights: {str(e)}"

    def chat_with_loan_assistant(self, context, user_query):
        """
        Generate a conversational response based on the loan application context and user query
        """
        system_prompt = """You are an experienced loan advisor specializing in credit risk, financial analysis, and lending strategy.
        Your role is to provide:
        - **Clear, structured financial advice**  
        - **Adaptive explanations** (simpler for beginners, technical for experts)  
        - **Actionable insights** (not just generic information)  
        - **Encouraging, supportive tone** (like a trusted financial coach)"""

        context_prompt = f"""📌 **LOAN APPLICATION CONTEXT**  
    {context}

    ### 💬 **User Query**:  
    ➡️ {user_query}

    ### 🎯 **Response Guidelines**
    - **If the user asks about rejection reasons**, break down risk factors with suggestions.  
    - **If they ask about approval likelihood**, give a percentage with justification.  
    - **If they seek advice on improvement**, offer **specific numbers** (e.g., "Reduce DTI by X%, increase savings by $Y").  
    - **If they ask a general question**, explain concepts simply but thoroughly.  
    - **Always be professional yet approachable.**"""

        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_prompt}
                ],
                max_tokens=750,
                temperature=0.6  # More engaging conversation
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"


def main():
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

                # Initialize chat history in session state if not exists
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []

                # Display chat history
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat input
                if prompt := st.chat_input("Ask a question about your loan application"):
                    # Add user message to chat history
                    st.session_state.chat_history.append({"role": "user", "content": prompt})

                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Generate context for chat
                    context = f"""Loan Application Details:
Prediction: {'APPROVED' if prediction == 0 else 'REJECTED'}
{json.dumps(user_data, indent=2)}
Feature Importance: {json.dumps(feature_importance, indent=2)}
Initial Insights: {initial_insights}"""

                    # Generate AI response
                    insights_generator = LoanInsightsGenerator(GROQ_API_KEY)
                    with st.chat_message("assistant"):
                        response = insights_generator.chat_with_loan_assistant(context, prompt)
                        st.markdown(response)

                    # Add AI response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"❌ Error during assessment: {e}")
                import traceback
                st.error(traceback.format_exc())
        else:
            st.error("Model is not loaded. Cannot perform prediction.")


if __name__ == "__main__":
    main()