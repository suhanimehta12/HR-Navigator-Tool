import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

st.set_page_config(page_title="Employee Recruitment Model", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "Upload Dataset"

def go_to(page_name):
    st.session_state.page = page_name

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
}

MODEL_PATH = "trained_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"


if "data" in st.session_state:
    df = st.session_state["data"]
    df.dropna(inplace=True)

    X = df.drop(["HiringDecision", "PersonalityScore"], axis=1)
    y = df["HiringDecision"]

    categorical_cols = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

if st.session_state.page == "Upload Dataset":
    st.title("Employee Recruitment Model")
    st.write("Upload your dataset.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        required_cols = [
            "Age",
            "Gender",
            "EducationLevel",
            "ExperienceYears",
            "PreviousCompanies",
            "DistanceFromCompany",
            "InterviewScore",
            "SkillScore",
            "RecruitmentStrategy",
            "HiringDecision",
            "PersonalityScore",
        ]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Dataset missing required columns: {missing}")
            st.stop()

        st.session_state["data"] = df
        st.success("Dataset uploaded successfully.")

        if st.button("Next"):
            go_to("Department Hiring")

if st.session_state.page == "Department Hiring":
    st.title("Candidate Selection for Hiring")

    if "data" not in st.session_state:
        st.warning("Please upload dataset first.")
        st.stop()

    df = st.session_state["data"]

    strategy_map = {
        "Technical Hiring (Engineering / IT)": 1,
        "Corporate Hiring (HR / Admin)": 2,
        "Sales & Marketing Hiring": 3
    }

    st.subheader("Hiring Requirements")

    selected_department = st.selectbox(
        "Department",
        list(strategy_map.keys())
    )

    min_experience = st.slider(
        "Minimum Years of Experience",
        0, int(df["ExperienceYears"].max()), 2
    )

    min_interview_score = st.slider(
        "Minimum Interview Score",
        0, 100, 40
    )

    min_skill_score = st.slider(
        "Minimum Skill Score",
        0, 100, 50
    )

    strategy_value = strategy_map[selected_department]

    dept_df = df[
        (df["RecruitmentStrategy"] == strategy_value) &
        (df["ExperienceYears"] >= min_experience) &
        (df["InterviewScore"] >= min_interview_score) &
        (df["SkillScore"] >= min_skill_score)
    ]

    hire_n = st.slider(
        "Number of candidates to hire",
        1, max(1, len(dept_df)),
        min(2, len(dept_df)) if len(dept_df) >= 2 else 1
    )

    display_df = dept_df.drop(
        columns=["DistanceFromCompany", "PersonalityScore"],
        errors="ignore"
    )

    st.subheader("Eligible Candidates")
    st.dataframe(
        display_df.rename(columns={
            "ExperienceYears": "Experience (Years)",
            "InterviewScore": "Interview Score",
            "SkillScore": "Skill Score",
            "PreviousCompanies": "Previous Companies"
        })
    )

    if st.button("Train Model"):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(label_encoder, ENCODER_PATH)

        st.success("Model trained and saved.")

    if st.button("Select Best Candidates"):
        model = joblib.load(MODEL_PATH)
        scaler_loaded = joblib.load(SCALER_PATH)

        X_all = dept_df.drop(
            columns=["HiringDecision", "PersonalityScore"],
            errors="ignore"
        )

        X_all = pd.get_dummies(X_all)

        for col in X.columns:
            if col not in X_all:
                X_all[col] = 0

        X_all = scaler_loaded.transform(X_all[X.columns])

        probs = model.predict_proba(X_all)[:, 1]

        dept_df = dept_df.copy()
        dept_df["HiringScore"] = probs

        best_candidates = dept_df.sort_values(
            "HiringScore", ascending=False
        ).head(hire_n)

        st.subheader("Top Selected Candidates")
        st.dataframe(
            best_candidates.drop(
                columns=["DistanceFromCompany", "PersonalityScore"],
                errors="ignore"
            )
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back"):
            go_to("Upload Dataset")
    with col2:
        if st.button("Next"):
            go_to("EDA")


if st.session_state.page == "EDA":
    st.title("Exploratory Data Analysis")

    df = st.session_state["data"]

    fig, ax = plt.subplots()
    hiring_counts = df["HiringDecision"].value_counts(normalize=True) * 100
    sns.barplot(x=hiring_counts.index, y=hiring_counts, ax=ax)
    ax.set_ylabel("Percentage")

    st.pyplot(fig)

    if st.button("Back"):
        go_to("Department Hiring")
    if st.button("Next"):
        go_to("Model Evaluation")

if st.session_state.page == "Model Evaluation":
    st.title("Model Evaluation & ROC Curves")

    results = []
    fig, ax = plt.subplots()

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
        results.append([name, acc, prec, rec, f1])

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.legend()
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    st.pyplot(fig)

    results_df = pd.DataFrame(
        results,
        columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
    )
    st.dataframe(results_df)

    if st.button("Back"):
        go_to("EDA")
