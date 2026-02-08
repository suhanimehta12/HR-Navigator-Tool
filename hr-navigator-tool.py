
import streamlit as st

st.set_page_config(page_title="HR NAVIGATOR", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F4F6F9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("HR NAVIGATOR")

st.markdown("""
Welcome to HR NAVIGATOR
 This app provides three powerful HR tools to help you manage employee data and make better decisions in your organization.
  Choose from the following modes to get started:

1. Employee Attrition: Helps to understand and predict the likelihood of employees leaving the company.
   This model helps you analyze employee turnover and make informed decisions to reduce attrition.



2. Employee Recruitment: This model helps you streamline your recruitment process by evaluating potential candidates and predicting their suitability for the job.



3. Employee Promotion: Analyze employee performance and potential, and predict the likelihood of promotions based on various factors.

Select the application you want to explore below by clicking on the corresponding link!
""")

app_choice = st.radio("Select an Application:", ["Employee Attrition Application", "Employee Recruitment Application", "Employee Promotion Application"])

if app_choice == "Employee Attrition Application":
    st.markdown("[Employee Attrition Application](https://blank-app-ztexj6vr8xi.streamlit.app/)")

elif app_choice == "Employee Recruitment Application":
    st.markdown("[Employee Recruitment Application](https://blank-app-r88p17uu5ea.streamlit.app/)")

elif app_choice == "Employee Promotion Application":  
    st.markdown("[Employee Promotion Application](https://blank-app-0xeqw2zp56na.streamlit.app/)")

