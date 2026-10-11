"""
Streamlit interface for the Wine Quality Predictor.

Run (after running train_model.py once):
    streamlit run app.py
"""

import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")


@st.cache_resource
def load_artifacts():
    model = joblib.load("outputs/wine_model.pkl")
    with open("outputs/feature_info.json") as f:
        info = json.load(f)
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    return model, info, metrics


try:
    model, info, metrics = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Please run `python train_model.py` first.")
    st.stop()

features = info["features"]

st.title("🍷 Wine Quality Predictor")
st.write("Adjust the chemical properties of a red wine and the model will predict "
         "its quality score (typically between 3 and 8).")

best = metrics[info["best_model"]]
st.caption(f"Model: {info['best_model']}  |  Test R² = {best['R2']:.2f}  |  "
           f"Average error = {best['MAE']:.2f} points")

values = {}
left, right = st.columns(2)
for i, feat in enumerate(features):
    lo, hi, mid = float(info["min"][feat]), float(info["max"][feat]), float(info["median"][feat])
    step = float(f"{(hi - lo) / 100:.2g}") or 0.01
    with (left if i % 2 == 0 else right):
        values[feat] = st.slider(feat.title(), min_value=lo, max_value=hi, value=mid, step=step)

if st.button("Predict Quality", type="primary"):
    row = pd.DataFrame([values], columns=features)
    score = float(model.predict(row)[0])

    st.subheader("Result")
    st.metric("Predicted quality score", f"{score:.2f} / 10")
    st.progress(min(max(score / 10, 0.0), 1.0))

    if score < 5.5:
        st.warning("Below average quality")
    elif score < 6.5:
        st.info("Average quality")
    else:
        st.success("Good quality")

with st.expander("Which features matter most?"):
    imp = pd.Series(info["importance"]).sort_values(ascending=False)
    st.bar_chart(imp)
