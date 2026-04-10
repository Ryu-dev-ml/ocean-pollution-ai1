import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Ocean Pollution AI", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f8fafc;
}
h1, h2, h3 {
    color: #0f172a;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- TITLE ----------------
st.title("🌊 Ocean Pollution Detection System")
st.markdown("AI-powered analysis using YOLOv8")

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

    # ---------------- LAYOUT ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🧠 Detection Output")
        st.image(annotated, use_container_width=True)

    # ---------------- RESULT ----------------
    boxes = results[0].boxes

    st.markdown("---")
    st.subheader("📊 Analysis Dashboard")

    if boxes is None or len(boxes) == 0:
        st.success("✅ Clean Water Detected")
    else:
        st.error("❌ Pollution Detected")

        # Score
        score = min(100, len(boxes) * 20)

        # ---------------- METRICS ----------------
        m1, m2 = st.columns(2)

        with m1:
            st.metric("Pollution Score", f"{score}/100")

        with m2:
            st.metric("Objects Detected", len(boxes))

        # ---------------- CLASS ANALYSIS ----------------
        classes = boxes.cls.cpu().numpy()
        names = model.names

        class_names = [names[int(c)] for c in classes]

        df = pd.DataFrame(class_names, columns=["Type"])

        count_df = df["Type"].value_counts().reset_index()
        count_df.columns = ["Pollution Type", "Count"]

        # ---------------- TABLE + CHART ----------------
        col3, col4 = st.columns(2)

        with col3:
            st.write("### 🧾 Detection Table")
            st.dataframe(count_df)

        with col4:
            st.write("### 📈 Distribution")
            st.bar_chart(count_df.set_index("Pollution Type"))

        # ---------------- REPORT ----------------
        st.markdown("### 📄 AI Report")
        st.info(f"""
        The system detected {len(boxes)} polluted regions in the water body.

        Pollution score is **{score}/100**, indicating {'HIGH' if score > 60 else 'MODERATE'} pollution level.

        Major detected types: {", ".join(set(class_names))}
        """)
