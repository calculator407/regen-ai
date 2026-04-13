import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Regen", page_icon="🩺", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("output/diq010_model_pipeline.joblib")

pipe = load_model()

st.title("Regen")
st.caption("Health risk prediction app")

with st.sidebar:
    st.header("Upload data")
    uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
    run_pred = st.button("Run prediction")

st.write("Upload a CSV containing the same feature columns used during training.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview")
    st.dataframe(df.head(), use_container_width=True)

    if run_pred:
        pred = pipe.predict(df)

        result = df.copy()
        result["prediction"] = pred

        if hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(df)
                if len(proba.shape) == 2:
                    for i in range(proba.shape[1]):
                        result[f"prob_class_{i}"] = proba[:, i]
            except:
                pass

        st.subheader("Predictions")
        st.dataframe(result, use_container_width=True)

        st.download_button(
            "Download predictions as CSV",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="regen_predictions.csv",
            mime="text/csv"
        )
else:
    st.info("Upload a CSV file to start.")
