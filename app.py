import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Ocean Pollution AI", layout="wide")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- TITLE ----------------
st.title("🌊 Ocean Pollution Detection System")
st.caption("AI-powered analysis using YOLOv8")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload Water Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Run model
    results = model.predict(image_np, conf=confidence)
    annotated = results[0].plot()

    # ---------------- IMAGE DISPLAY ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Input Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🧠 Detection Output")
        st.image(annotated, use_container_width=True)

    # ---------------- ANALYSIS ----------------
    st.divider()
    st.subheader("📊 Analysis")

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        st.success("✅ Clean Water Detected")
    else:
        st.error("❌ Pollution Detected")

        score = min(100, len(boxes) * 20)

        # Metrics
        col3, col4, col5 = st.columns(3)

        with col3:
            st.metric("Pollution Score", f"{score}/100")

        with col4:
            st.metric("Objects Detected", len(boxes))

        with col5:
            st.metric("Severity", "High" if score > 60 else "Moderate")

        # Class analysis
        classes = boxes.cls.cpu().numpy()
        names = model.names

        class_names = [names[int(c)] for c in classes]

        df = pd.DataFrame(class_names, columns=["Type"])
        count_df = df["Type"].value_counts().reset_index()
        count_df.columns = ["Pollution Type", "Count"]

        col6, col7 = st.columns(2)

        with col6:
            st.subheader("🧾 Detection Table")
            st.dataframe(count_df, use_container_width=True)

        with col7:
            st.subheader("📈 Distribution")
            st.bar_chart(count_df.set_index("Pollution Type"))

        # Report
        st.subheader("📄 AI Report")
        st.info(f"""
        The system detected **{len(boxes)} polluted regions**.

        Pollution score: **{score}/100** → {'HIGH' if score > 60 else 'MODERATE'} level.

        Detected types: {", ".join(set(class_names))}
        """)
