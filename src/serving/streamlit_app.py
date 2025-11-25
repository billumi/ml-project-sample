
import streamlit as st
import pickle, os, numpy as np

MODEL_PATH = os.environ.get("MODEL_PATH", "models/saved/model.pkl")

st.title("ML Toolkit - Streamlit Demo")

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    st.success("Model loaded")
else:
    st.warning("Model not found - please train and save to models/saved/model.pkl")

st.write("Enter features as comma separated values:")
s = st.text_input("features", "5.1,3.5,1.4,0.2")
if st.button("Predict"):
    try:
        arr = np.array([float(x.strip()) for x in s.split(",")]).reshape(1,-1)
        pred = model.predict(arr).tolist()
        st.write("Prediction:", pred)
    except Exception as e:
        st.error(str(e))
