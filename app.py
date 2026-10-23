
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Student Success AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI-Powered Student Success Dashboard")
st.write(
    "Explore student performance data and identify "
    "students who may benefit from additional support."
)

# Load saved files
@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("student_data.csv")

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Could not load model or dataset: {e}")
    st.info(
        "Keep app.py, best_model.joblib, and "
        "student_data.csv in the same folder."
    )
    st.stop()

# Prepare the same features used during training
excluded = ["G1", "G2", "G3", "at_risk"]
X = df.drop(
    columns=[c for c in excluded if c in df.columns]
)

st.subheader("📊 Dataset Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Total student records", len(df))
c2.metric("Input features", X.shape[1])
c3.metric(
    "Average final grade",
    f"{df['G3'].mean():.2f}/20"
    if "G3" in df.columns else "N/A"
)

with st.expander("View dataset"):
    st.dataframe(df.head(20), use_container_width=True)

st.subheader("📈 Explore the Data")

if "G3" in df.columns:
    fig, ax = plt.subplots()
    ax.hist(df["G3"], bins=11, edgecolor="black")
    ax.set_title("Final Grade Distribution")
    ax.set_xlabel("Final grade")
    ax.set_ylabel("Number of students")
    st.pyplot(fig)
    plt.close(fig)

if "studytime" in df.columns and "G3" in df.columns:
    study_summary = df.groupby("studytime")["G3"].mean()
    st.bar_chart(study_summary)

st.subheader("🤖 Student Risk Prediction")
st.write(
    "Enter student information to estimate the model's "
    "predicted risk. This is an experimental educational "
    "prototype, not a definitive assessment."
)

with st.form("student_form"):
    inputs = {}

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            median_value = float(X[col].median())
            min_value = float(X[col].min())
            max_value = float(X[col].max())

            inputs[col] = st.number_input(
                col,
                min_value=min_value,
                max_value=max_value,
                value=min(max(median_value, min_value), max_value)
            )
        else:
            choices = sorted(
                X[col].dropna().astype(str).unique().tolist()
            )

            if not choices:
                choices = ["Unknown"]

            inputs[col] = st.selectbox(col, choices)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    input_df = pd.DataFrame([inputs], columns=X.columns)

    try:
        prediction = int(model.predict(input_df)[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)
            risk_index = classes.index(1)
            risk_probability = float(probabilities[risk_index])

            st.metric(
                "Estimated risk probability",
                f"{risk_probability:.1%}"
            )

        if prediction == 1:
            st.warning(
                "The model flags this profile as potentially at risk."
            )
        else:
            st.success(
                "The model does not flag this profile as at risk."
            )

        st.caption(
            "Predictions are uncertain and should not be used "
            "to label or penalize students. Seek human review "
            "and appropriate support."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption(
    "Project by M.Sc. Data Science | "
    "Model trained using the UCI Student Performance dataset."
)
