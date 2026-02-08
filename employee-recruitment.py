import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


st.set_page_config(page_title="Employee Recruitment Prediction", layout="wide")


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
    "This app predicts hiring decisions based on candidate features using various classification models. "
    "Upload a dataset, analyze data, and compare model performances.</div>",
    unsafe_allow_html=True
)


page = st.sidebar.radio("Navigation", ["Upload Dataset", "Predict Hiring Decision", "EDA", "Model Performance", "Classification Report"])


models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}


if page == "Upload Dataset":
    st.title("Upload the  Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview")
        st.dataframe(df.head())

        if "HiringDecision" not in df.columns:
            st.error("Dataset is missing 'HiringDecision' column.")
            st.stop()

 
        st.session_state["data"] = df


if "data" in st.session_state:
    df = st.session_state["data"]
    df.dropna(inplace=True)

    X = df.drop("HiringDecision", axis=1)
    y = df["HiringDecision"]

    categorical_cols = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


if page == "Predict Hiring Decision":
    st.title("Predict Hiring Decision")

    st.write("Enter Candidate Details")
    user_input = {}
    for feature in df.drop(columns=["HiringDecision"]).columns:
        unique_values = df[feature].unique().tolist()
        user_input[feature] = st.selectbox(f"Select value for {feature}", unique_values)

    user_df = pd.DataFrame([user_input])
    user_df = pd.get_dummies(user_df, columns=categorical_cols, drop_first=True)


    for col in X.columns:
        if col not in user_df:
            user_df[col] = 0

    user_df = scaler.transform(user_df)


    st.write("Select a Model for Prediction")
    selected_model_name = st.selectbox("Choose a Model", list(models.keys()))
    selected_model = models[selected_model_name]


    if st.button("Predict Hiring Decision"):
        selected_model.fit(X_train, y_train)
        prediction = selected_model.predict(user_df)
        pred_class = label_encoder.inverse_transform(prediction)


        st.write(f"Predicted Hiring Decision: {pred_class[0]}**")

        st.markdown("""
            <div class='report-box'>
            <b>Prediction Meaning:</b><br>
            - **0** = Not Hired <br>
            - **1** = Hired
            </div>
        """, unsafe_allow_html=True)

if page == "EDA":
    st.title("Exploratory Data Analysis")

    if "data" in st.session_state:
        st.write("Hiring Decision Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        hiring_counts = df["HiringDecision"].value_counts(normalize=True) * 100
        sns.barplot(x=hiring_counts.index, y=hiring_counts, palette="coolwarm", ax=ax)
        ax.set_ylabel("Percentage of Candidates")
        ax.set_xticklabels(["Not Hired", "Hired"])
        for i, v in enumerate(hiring_counts):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=12)
        st.pyplot(fig)


if page == "Model Performance":
    st.title("Model Performance Comparison")

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


    df_results = pd.DataFrame({
        "Model": accuracy_results.keys(),
        "Accuracy (%)": [f"{acc:.2f}%" for acc in accuracy_results.values()],
        "Training Time (s)": training_times.values(),
        "Testing Time (s)": testing_times.values()
    })
    st.dataframe(df_results)


    st.write("Model Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(accuracy_results.keys()), y=list(accuracy_results.values()), palette="coolwarm", ax=ax)
    ax.set_ylabel("Accuracy (%)")
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
    st.dataframe(report_df)

