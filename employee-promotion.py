import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


st.set_page_config(page_title="Employee Promotion Prediction", layout="wide")
st.title("Employee Promotion Prediction")


page = st.sidebar.radio("Select a page", ["Upload Dataset", "Prediction", "EDA", "Model Performance", "Classification Report"])


if page == "Upload Dataset":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read the dataset
            df = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:")
            st.dataframe(df.head())

            st.write("Dataset Info:")
            st.write(df.describe(include="all"))
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

            st.write("Missing Values:")
            st.write(df.isna().sum())

            # Store dataset in session state for use in other pages
            st.session_state.df = df

        except Exception as e:
            st.error(f"An error occurred: {e}")


elif page == "Prediction":
    if 'df' in st.session_state:
        df = st.session_state.df
        # Preprocessing
        df.dropna(inplace=True)

        selected_features = ['employee_id', 'department', 'region', 'education', 'gender', 'recruitment_channel',
                             'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'awards_won',
                             'avg_training_score']

        categorical_features = df[selected_features].select_dtypes(include=['object']).columns.tolist()

        encoder = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
            remainder='passthrough'
        )


        X = df[selected_features]
        y = df['is_promoted']

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        X_encoded = encoder.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        user_data = {}
        for feature in selected_features:
            if feature in categorical_features:
                options = df[feature].unique().tolist()
                value = st.selectbox(f"Select value for {feature}", options)
            else:
                value = st.slider(f"Select value for {feature}", int(df[feature].min()), int(df[feature].max()), int(df[feature].mean()))
            user_data[feature] = [value]

        selected_model_name = st.selectbox("Select a model", ["Logistic Regression", "Random Forest", "Decision Tree", "Gradient Boosting"])
        models = {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        selected_model = models[selected_model_name]

        if st.button("Train Model and Predict Promotion"):
            # Ensure the model is trained before prediction
            st.write(f"Training the {selected_model_name} model...")
            

            selected_model.fit(X_train, y_train)


            user_df = pd.DataFrame(user_data)
            user_input_transformed = encoder.transform(user_df)
            prediction = selected_model.predict(user_input_transformed)
            pred_class = label_encoder.inverse_transform(prediction)

            st.write(f"Predicted Promotion: {pred_class[0]}")

            if pred_class[0] == 0:
                st.markdown("""
                0 indicates that the employee is not promoted.""")
            else:
                st.markdown("""
                1indicates that the employee is likely to be promoted.""")


elif page == "EDA":
    if 'df' in st.session_state:
        df = st.session_state.df
        st.write("Exploratory Data Analysis (EDA)")

      
        st.write("Promotions by Department")
        fig, ax = plt.subplots()
        sns.countplot(x='department', hue='is_promoted', data=df, palette='coolwarm')
        plt.title("Promotions by Department")
        plt.xticks(rotation=45)
        st.pyplot(fig)


elif page == "Model Performance":
    if 'df' in st.session_state:
        df = st.session_state.df
        df.dropna(inplace=True)

        selected_features = ['employee_id', 'department', 'region', 'education', 'gender', 'recruitment_channel',
                             'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'awards_won',
                             'avg_training_score']

        categorical_features = df[selected_features].select_dtypes(include=['object']).columns.tolist()

        encoder = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
            remainder='passthrough'
        )

        X = df[selected_features]
        y = df['is_promoted']

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        X_encoded = encoder.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Random Forest": RandomForestClassifier(),
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

            accuracy_results[model_name] = accuracy_score(y_test, y_pred) * 100

        accuracy_df = pd.DataFrame({
            "Model": list(accuracy_results.keys()),
            "Accuracy (%)": list(accuracy_results.values()),
            "Training Time (s)": list(training_times.values()),
            "Testing Time (s)": list(testing_times.values())
        })
        st.table(accuracy_df)

        # Accuracy comparison bar plot
        st.write("Model Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=list(accuracy_results.keys()), y=list(accuracy_results.values()), palette="coolwarm")
        plt.ylabel("Accuracy (%)")
        plt.title("Model Accuracy Comparison")
        st.pyplot(fig)

elif page == "Classification Report":
    if 'df' in st.session_state:
        df = st.session_state.df
        df.dropna(inplace=True)

        selected_features = ['employee_id', 'department', 'region', 'education', 'gender', 'recruitment_channel',
                             'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'awards_won',
                             'avg_training_score']

        categorical_features = df[selected_features].select_dtypes(include=['object']).columns.tolist()

        encoder = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
            remainder='passthrough'
        )

        X = df[selected_features]
        y = df['is_promoted']

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        X_encoded = encoder.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)


        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        classification_df = pd.DataFrame(classification_rep).transpose()
        st.write("Classification Report:")
        st.table(classification_df)
