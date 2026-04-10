import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO("best.pt")

st.title("🌊 Ocean Pollution Detection System")

uploaded_file = st.file_uploader("Upload Water Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    results = model.predict(image, conf=0.25)
    result_img = results[0].plot()

    st.image(result_img, caption="Detection Result", use_container_width=True)

    if len(results[0].boxes) == 0:
        st.success("Clean Water ✅")
    else:
        st.error("Polluted Water ❌")
        score = min(100, len(results[0].boxes) * 20)
        st.write(f"Pollution Score: {score}")
