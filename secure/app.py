import app as st
import pandas as pd
import pickle

# Load the trained model pipeline
MODEL_PATH = "C:\SatyamsFolder\projects\ML\Credit_Underwriting\pipeline_1.pkl"

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure 'Model_pipeline.pkl' is in the correct directory.")
    model = None

# Streamlit App UI
st.title("🎯 Loan Approval Predicton Application")
st.write("This system analyzes applicant data to predict the likelihood of loan approval.")

st.markdown("### 📋 Applicant Information")
col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Applicant's current age")
    home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    loan_amnt = st.number_input("Requested Loan Amount ($)", min_value=0, value=10000, step=500)
    loan_intent = st.selectbox("Loan Purpose", ["MEDICAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT", "VENTURE", "PERSONAL", "EDUCATION"])
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=10, max_value=60)

with col2:
    person_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    person_emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
    loan_int_rate = st.slider("Interest Rate (%)", min_value=0.0, max_value=30.0, value=5.0, step=0.1)
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])

# cb_person_default_on_file is set permanently as "N"
cb_person_default_on_file = "N"

if st.button("Analyze Loan Approval") and model:
    try:
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

        prediction = model.predict(user_input)[0]

        if prediction == 0:
            st.balloons()
            st.success("✅ Loan is likely to get approved: The applicantion shows good indicators for loan approval.")
        else:
            st.warning("⚠️ Application likely to get rejected: The applicantion shows elevated chance of rejection.")
    except Exception as e:
        st.error(f"❌ Error during assessment: {e}")
