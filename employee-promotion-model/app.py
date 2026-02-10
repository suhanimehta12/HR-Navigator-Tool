import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Employee Promotion Model", layout="wide")
st.title("Employee Promotion Model")


if "page" not in st.session_state:
    st.session_state.page = 0

PAGES = ["Upload", "Promotion", "EDA", "Evaluation"]

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1


def prepare_data(df):
    features = [
        'employee_id', 'department', 'region', 'education', 'gender',
        'recruitment_channel', 'no_of_trainings', 'age',
        'previous_year_rating', 'length_of_service',
        'awards_won', 'avg_training_score'
    ]

    cat_cols = df[features].select_dtypes(include='object').columns

    encoder = ColumnTransformer(
        [('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
        remainder='passthrough'
    )

    X = encoder.fit_transform(df[features])
    y = LabelEncoder().fit_transform(df['is_promoted'])

    return X, y, encoder

if PAGES[st.session_state.page] == "Upload":
    st.subheader("Upload Your Dataset")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file).dropna()
        st.session_state.df = df
        st.success("Dataset uploaded successfully")
        st.dataframe(df.head())

    st.button("Next", on_click=next_page)

elif PAGES[st.session_state.page] == "Promotion":
    df = st.session_state.df

    X, y, encoder = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    scores = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        scores[name] = accuracy_score(y_test, preds)
        trained_models[name] = model

   
    best_model_name = max(scores, key=scores.get)
    best_model = trained_models[best_model_name]

    #st.info(f"Using best model: **{best_model_name}**")

    df['promotion_probability'] = best_model.predict_proba(X)[:, 1]

    #st.subheader("Department-wise Promotion Allocation")

    dept = st.selectbox("Select Department", df['department'].unique())
    seats = st.number_input("Promotion seats", min_value=1, max_value=10, value=2)

    dept_df = df[df['department'] == dept].sort_values(
        by='promotion_probability', ascending=False
    )

    promoted = dept_df.head(seats)

    st.subheader("Promoted Employees")
    st.dataframe(
        promoted[
            ['employee_id', 'department', 'promotion_probability']
        ]
    )

    col1, col2 = st.columns(2)
    col1.button("⬅ Back", on_click=prev_page)
    col2.button("Next ➡", on_click=next_page)

elif PAGES[st.session_state.page] == "EDA":
    df = st.session_state.df

    st.subheader("Exploratory Data Analysis")

    fig, ax = plt.subplots()
    sns.countplot(x='department', hue='is_promoted', data=df)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.boxplot(x='is_promoted', y='avg_training_score', data=df)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.boxplot(x='is_promoted', y='length_of_service', data=df)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.button("Back", on_click=prev_page)
    col2.button("Next", on_click=next_page)

elif PAGES[st.session_state.page] == "Evaluation":

    st.subheader("Model Evaluation & Comparison")

    df = st.session_state.df.dropna()


    features = [
        'employee_id', 'department', 'region', 'education', 'gender',
        'recruitment_channel', 'no_of_trainings', 'age',
        'previous_year_rating', 'length_of_service',
        'awards_won', 'avg_training_score'
    ]

    cat_cols = df[features].select_dtypes(include='object').columns

    encoder = ColumnTransformer(
        [('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
        remainder='passthrough'
    )

    X = encoder.fit_transform(df[features])
    y = LabelEncoder().fit_transform(df['is_promoted'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    metrics = []

    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        metrics.append([
            name, acc, prec, rec, f1, roc_auc
        ])

        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    st.pyplot(plt)

    metrics_df = pd.DataFrame(
        metrics,
        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    )

    st.subheader("Evaluation Metrics Comparison")
    st.dataframe(metrics_df.style.format({
        "Accuracy": "{:.2f}",
        "Precision": "{:.2f}",
        "Recall": "{:.2f}",
        "F1 Score": "{:.2f}",
        "ROC-AUC": "{:.2f}"
    }))

    st.button("Back", on_click=prev_page)

