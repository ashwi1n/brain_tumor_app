import streamlit as st
from PIL import Image
from utils import predict_mri

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection from MRI")
st.write("Upload an MRI scan to check if it has a brain tumor.")

uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI", use_container_width=True)
    
    with st.spinner("Analyzing Image..."):
        result = predict_mri(img)
    
    st.success(f"Prediction: **{result}**")
