import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")


st.markdown("""
    <style>
    body { background-color: #f5f5f5; }
    .sidebar .sidebar-content { background-color: #2E3B55; color: white; }
    h1, h2, h3 { font-family: 'Arial Black', sans-serif; color: #2E3B55; }
    .report-box { background-color: #fafafa; padding: 10px; border-radius: 10px; border: 1px solid #ddd; }
    </style>
""", unsafe_allow_html=True)


st.sidebar.markdown(
    "<div class='report-box'><b>About This Model:</b><br>"
    "This app predicts employee attrition using various classification models. "
    "It allows users to upload a dataset, analyze data, and compare model performances.</div>",
    unsafe_allow_html=True
)


page = st.sidebar.radio("Navigation", ["Upload Dataset", "Predict Attrition", "EDA", "Model Performance", "Classification Report"])


if page == "Upload Dataset":
    st.title("Upload the Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview")
        st.dataframe(df.head())

        # Save dataset for later use
        st.session_state["data"] = df


if "data" in st.session_state:
    df = st.session_state["data"]

 
    df.dropna(inplace=True)
    drop_columns = ['EmployeeNumber', 'StockOptionLevel']
    df.drop(columns=[col for col in drop_columns if col in df.columns], axis=1, inplace=True)

    selected_features = ['Department', 'JobRole', 'MaritalStatus', 'OverTime', 'JobSatisfaction', 'Age']
    categorical_features = df[selected_features].select_dtypes(include=['object']).columns.tolist()

    encoder = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )

    X = df[selected_features]
    y = df['Attrition']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_encoded = encoder.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


if page == "Predict Attrition":
    st.title("Predict Employee Attrition")


    model_choice = st.selectbox("Select a Model", ["Logistic Regression", "Random Forest", "Support Vector Machine", "Decision Tree", "Gradient Boosting"])


    department = st.selectbox("Department", df["Department"].unique())
    job_role = st.selectbox("Job Role", df["JobRole"].unique())
    marital_status = st.selectbox("Marital Status", df["MaritalStatus"].unique())
    overtime = st.selectbox("OverTime", df["OverTime"].unique())
    job_satisfaction = st.slider("Job Satisfaction", min_value=int(df["JobSatisfaction"].min()), max_value=int(df["JobSatisfaction"].max()), value=3)
    age = st.slider("Age", min_value=int(df["Age"].min()), max_value=int(df["Age"].max()), value=30)


    input_data = pd.DataFrame([[department, job_role, marital_status, overtime, job_satisfaction, age]], columns=selected_features)
    input_encoded = encoder.transform(input_data)


    model_mapping = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    model = model_mapping[model_choice]


    model.fit(X_train, y_train)

    if st.button("Predict"):
        prediction = model.predict(input_encoded)
        result = "Yes" if prediction[0] == 1 else "No"
        st.write(f"Prediction: Employee Attrition - **{result}**")


if page == "EDA":
    st.title("Exploratory Data Analysis")

    if "data" in st.session_state:
        # Attrition Count Plot
        st.write("Overall Attrition Percentage")
        fig, ax = plt.subplots(figsize=(6, 4))
        attrition_counts = df["Attrition"].value_counts(normalize=True) * 100
        sns.barplot(x=attrition_counts.index, y=attrition_counts, palette="coolwarm", ax=ax)
        ax.set_ylabel("Percentage of Employees")
        ax.set_xticklabels(["No Attrition", "Yes Attrition"])
        for i, v in enumerate(attrition_counts):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=12)
        st.pyplot(fig)


if page == "Model Performance":
    st.title("Model Performance Comparison")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    accuracy_results = {}
    training_times = {}
    testing_times = {}

    for model_name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_times[model_name] = time.time() - start_time

        start_time = time.time()
        y_pred = model.predict(X_test)
        testing_times[model_name] = time.time() - start_time

        accuracy_results[model_name] = accuracy_score(y_test, y_pred) * 100  # Convert to percentage


    st.write("Model Performance Metrics")
    df_results = pd.DataFrame({
        "Model": accuracy_results.keys(),
        "Accuracy (%)": [f"{acc:.2f}%" for acc in accuracy_results.values()],  # Format as percentage
        "Training Time (s)": training_times.values(),
        "Testing Time (s)": testing_times.values()
    })
    st.dataframe(df_results)

    st.write("Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(accuracy_results.keys()), y=list(accuracy_results.values()), palette="coolwarm", ax=ax)
    ax.set_ylabel("Accuracy (%)")  # Update label to show percentage
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    for i, v in enumerate(accuracy_results.values()):
        ax.text(i, v + 1, f"{v:.2f}%", ha="center", fontsize=12)

    st.pyplot(fig)

if page == "Classification Report":
    st.title("Classification Report")

    selected_model = RandomForestClassifier()
    selected_model.fit(X_train, y_train)
    y_pred = selected_model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write("### Classification Report")
    st.dataframe(report_df)


