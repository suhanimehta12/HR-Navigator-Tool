import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from fpdf import FPDF
import io

st.set_page_config(page_title="Employee Attrition Model", layout="wide")

st.markdown("""
<style>
h1, h2, h3 { color: #2E3B55; font-family: Arial Black; }
.box { background: #fafafa; padding: 15px; border-radius: 12px; border: 1px solid #ddd; }
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = 1

def next_page():
    st.session_state.page += 1

def back_page():
    st.session_state.page -= 1

if st.session_state.page == 1:
    st.title(" Page 1: Upload Employee Dataset")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["data"] = df
        st.success("Dataset Uploaded Successfully!")
        st.dataframe(df.head())
        st.button("Next ", on_click=next_page)


if "data" in st.session_state:
    df = st.session_state["data"]
    df.dropna(inplace=True)
    
    drop_columns = ['EmployeeNumber', 'StockOptionLevel']
    df.drop(columns=[c for c in drop_columns if c in df.columns], inplace=True)

    selected_features = ['Department', 'JobRole', 'MaritalStatus', 'OverTime', 'JobSatisfaction', 'Age']

    categorical_features = df[selected_features].select_dtypes(include=['object']).columns.tolist()

    encoder = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
        remainder="passthrough"
    )

    X = df[selected_features]
    y = df["Attrition"]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X_encoded = encoder.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

if st.session_state.page == 2:
    st.title("Page 2: HR Attrition Decision Dashboard")
    st.markdown("<div class='box'>Enter employee details below:</div>", unsafe_allow_html=True)

    department = st.selectbox("Department", df["Department"].unique())
    job_role = st.selectbox("Job Role", df["JobRole"].unique())
    marital_status = st.selectbox("Marital Status", df["MaritalStatus"].unique())
    overtime = st.selectbox("OverTime", df["OverTime"].unique())
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    age = st.slider("Age", 18, 60, 30)

    input_data = pd.DataFrame([[department, job_role, marital_status, overtime, job_satisfaction, age]], columns=selected_features)
    input_encoded = encoder.transform(input_data)


    best_model = None
    best_acc = 0
    best_name = ""
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
    st.info(f" Best Model Automatically Selected: **{best_name}**")

    if st.button("Predict Attrition Risk"):
        prob = best_model.predict_proba(input_encoded)[0][1]
        risk_percent = prob * 100
        st.write(f" Attrition Risk Score: **{risk_percent:.2f}%**")

        if risk_percent > 80:
            st.error("Employee may leave within ~30 days")
        elif risk_percent > 50:
            st.warning("Employee may leave within ~60 days")
        else:
            st.success("Employee likely to stay for 90+ days")

       
        dept_df = df[df["Department"] == department]
        dept_encoded = encoder.transform(dept_df[selected_features])
        probs_all = best_model.predict_proba(dept_encoded)[:, 1]
        top_idx = np.argsort(probs_all)[-3:]

        #st.write(f"Top 3 Employees Most Likely to Leave in {department} Department")
        #top3 = dept_df.iloc[top_idx]
        #st.dataframe(top3)

        
        fire_employee = dept_df.iloc[top_idx[-1]]
        st.error(f"Only 1 position available → Fire the Highest Risk Employee: **{fire_employee['JobRole']} ({fire_employee['Department']})**")

  
        if department == "IT":
            st.write(" IT Department High Risk Employees")
            st.dataframe(df[df["Department"]=="IT"].head(5))

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "HR Report - Fired Employee", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.ln(10)
        pdf.cell(0, 8, f"Employee Name / Role: {fire_employee['JobRole']}", ln=True)
        pdf.cell(0, 8, f"Department: {fire_employee['Department']}", ln=True)
        pdf.cell(0, 8, f"Age: {fire_employee['Age']}", ln=True)
        pdf.cell(0, 8, f"Marital Status: {fire_employee['MaritalStatus']}", ln=True)
        pdf.cell(0, 8, f"OverTime: {fire_employee['OverTime']}", ln=True)
        pdf.cell(0, 8, f"Job Satisfaction: {fire_employee['JobSatisfaction']}", ln=True)
        pdf.cell(0, 8, f"Attrition Risk Score: {risk_percent:.2f}%", ln=True)
        pdf.ln(10)
        pdf.multi_cell(0, 8, "HR Recommendation: Employee is at highest risk and should be considered for termination based on company policy.")

 
        pdf_bytes = io.BytesIO(pdf.output(dest='S').encode('latin-1'))

        st.download_button(
            label=" Download HR Report PDF",
            data=pdf_bytes,
            file_name=f"HR_Report_{fire_employee['JobRole']}.pdf",
            mime="application/pdf"
        )

    col1, col2 = st.columns(2)
    col1.button("Back", on_click=back_page)
    col2.button("Next", on_click=next_page)

if st.session_state.page == 3:
    st.title("Page 3: Exploratory Data Analysis")
    st.write("Attrition Percentage")
    fig, ax = plt.subplots()
    attrition_counts = df["Attrition"].value_counts(normalize=True) * 100
    sns.barplot(x=attrition_counts.index, y=attrition_counts.values, ax=ax)
    st.pyplot(fig)

    st.write("Department Wise Attrition")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x="Department", hue="Attrition", ax=ax2)
    st.pyplot(fig2)

    col1, col2 = st.columns(2)
    col1.button("Back", on_click=back_page)
    col2.button("Next", on_click=next_page)

if st.session_state.page == 4:
    st.title("Page 4: Model Evaluation Metrics")

  
    best_model = None
    best_acc = 0
    best_name = ""
    model_results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

        report = classification_report(y_test, y_pred, output_dict=True)
        model_results[name] = {
            "Accuracy": acc,
            "Precision": report['weighted avg']['precision'],
            "Recall": report['weighted avg']['recall'],
            "F1 Score": report['weighted avg']['f1-score']
        }

    st.write(f"Best Model: {best_name}")

    metrics_df = pd.DataFrame(model_results).T
    metrics_df = metrics_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']]*100
    st.dataframe(metrics_df.style.format("{:.2f}"))

    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    st.write("Confusion Matrix (Best Model)")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    
    st.write("ROC Curve (Best Model)")
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:,1]
    else:
        y_prob = best_model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    ax2.plot([0,1],[0,1],'--', color='gray')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

    st.button("Back", on_click=back_page)

