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

st.write("Upload a CSV with the same feature columns used during training.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("Predict"):
        preds = pipe.predict(df)
        result = df.copy()
        result["prediction"] = preds

        st.subheader("Results")
        st.dataframe(result, use_container_width=True)

        st.download_button(
            "Download predictions",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="regen_predictions.csv",
            mime="text/csv"
        )
else:
    st.info("Upload a CSV to start.")
