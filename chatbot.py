import streamlit as st
from backend import LoanInsightsGenerator

def initialize_chat_session():
    """Initialize chat session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def display_chat_history():
    """Display existing chat messages"""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_chat_interaction(context, groq_api_key):
    """Handle chat input and generate AI responses"""
    # Chat input
    if prompt := st.chat_input("Ask a question about your loan application"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        insights_generator = LoanInsightsGenerator(groq_api_key)
        with st.chat_message("assistant"):
            response = insights_generator.chat_with_loan_assistant(context, prompt)
            st.markdown(response)

        # Add AI response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})