import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import io

# ---------------- CONFIG ----------------

st.set_page_config(page_title="Ocean Pollution AI", layout="wide")

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- HEADER ----------------

st.markdown("""
<h1 style='text-align: center; color: #00BFFF;'>
🌊 AI Marine Pollution Monitoring System
</h1>
""", unsafe_allow_html=True)

st.caption("Real-time Ocean Pollution Detection using YOLOv8")

# ---------------- SIDEBAR ----------------

st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.05, 0.5, 0.15)

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

    if boxes is None or len(boxes.cls) == 0:
        st.success("🌊 Clean Water Detected")
        st.metric("Pollution Score", "0/100")
    else:
        # Extract classes
        classes = boxes.cls.cpu().numpy()
        names = model.names
        class_names = [names[int(c)] for c in classes]

        # Weighted scoring
        weights = {"plastic": 30, "oil": 40, "algae": 20}
        score = sum([weights.get(name, 10) for name in class_names])
        score = min(100, score)

        # Severity
        if score < 40:
            severity = "Low"
        elif score < 70:
            severity = "Moderate"
        else:
            severity = "High"

        # Status display
        if score < 40:
            st.info("🟢 Low Pollution Level")
        elif score < 70:
            st.warning("🟡 Moderate Pollution Level")
        else:
            st.error("🔴 High Pollution Level")

        # Metrics
        col3, col4, col5 = st.columns(3)

        with col3:
            st.metric("Pollution Score", f"{score}/100")

        with col4:
            st.metric("Objects Detected", len(class_names))

        with col5:
            st.metric("Severity", severity)

        # Data table
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
        The system detected **{len(class_names)} polluted regions**.

        Pollution score: **{score}/100** → **{severity.upper()}** level.

        Detected types: {", ".join(count_df["Pollution Type"].tolist())}
        """)

        # Download result
        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="PNG")

        st.download_button(
            "📥 Download Result Image",
            buf.getvalue(),
            "result.png",
            "image/png"
        )

