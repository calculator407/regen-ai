
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

uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("Predict"):
        pred = pipe.predict(df)
        result = df.copy()
        result["prediction"] = pred
        st.dataframe(result, use_container_width=True)
        st.download_button(
            "Download predictions",
            result.to_csv(index=False).encode("utf-8"),
            file_name="regen_predictions.csv",
            mime="text/csv"
        )
else:
    st.info("Upload a CSV to start.")
