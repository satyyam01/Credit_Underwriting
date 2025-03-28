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
        # (Keep the existing method from the original code)
        system_prompt = """You are a senior loan underwriter with 20 years of experience in credit risk assessment..."""

        # Rest of the method remains the same as in the original code
        context_prompt = f"""### 🏦 Loan Application Risk Assessment..."""

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
            return f"Error generating insights: {str(e)}"

    def chat_with_loan_assistant(self, context, user_query):
        # (Keep the existing method from the original code)
        system_prompt = """You are an experienced loan advisor specializing in credit risk..."""

        context_prompt = f"""📌 **LOAN APPLICATION CONTEXT**  ..."""

        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_prompt}
                ],
                max_tokens=750,
                temperature=0.6
            )
            return response.choices[0].message.content
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