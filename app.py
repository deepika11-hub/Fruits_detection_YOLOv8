# ============================================================
# 🍎 Streamlit Fruit Detection App (Bujji Edition 💖)
# ============================================================
import os
os.system('apt-get update -y && apt-get install -y libgl1-mesa-glx')

import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os
import base64

# ============================================================
# 🌈 PAGE SETUP
# ============================================================
st.set_page_config(page_title="🍉Fruit Detection", page_icon="🍎", layout="centered")

# ============================================================
# 🍋 BACKGROUND SETUP (your background path here 👇)
# Example: "static/background/fresh.jpg"
# ============================================================
def add_bg_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            h1, h2, h3, p, label {{
                color: #fff !important;
                text-shadow: 0 0 10px rgba(0, 0, 0, 0.8);
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# 👇 Background image file path — update it to your file
add_bg_image("fruit_bg.jpg")

# ============================================================
# 🍎 LOAD YOLO MODEL
# ============================================================
model_path = "best.pt"
if not os.path.exists(model_path):
    st.error("❌ best.pt not found! Please place it in the app folder.")
else:
    model = YOLO(model_path)

# ============================================================
# 🍊 APP TITLE
# ============================================================
st.title("🍓Fruit Detection with YOLOv8🍊")
st.write("Upload or capture an image — detect fruits with bounding boxes and confidence! 🍇")

# ============================================================
# 📸 TABS (Upload / Webcam)
# ============================================================
# === Custom CSS for bold tab labels ===
st.markdown("""
    <style>
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-weight: 900 !important;    /* makes text bold */
        font-size: 20px !important;     /* optional: increase size */
        color: #ff1493 !important;      /* optional: pink color like your style 💖 */
    }
    </style>
    """, unsafe_allow_html=True)

# === Bold Tabs ===
tab1, tab2 = st.tabs(["📁 Upload Fruit Image", "🎥 Detect with Webcam"])

# ============================================================
# 📁 UPLOAD TAB
# ============================================================
with tab1:
    uploaded = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded)
        img_np = np.array(img)

        st.image(img, caption="📸 Uploaded Image", width=400)

        with st.spinner("Detecting fruits..."):
            results = model.predict(img_np, conf=0.5)
            result = results[0]

            annotated = img_np.copy()
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                text = f"{label} ({conf*100:.1f}%)"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (x1, y1 - 25), (x1 + w, y1), (255, 0, 0), -1)
                cv2.putText(annotated, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated, caption="✅ Detected Fruits", width=550)

            if len(result.boxes) > 0:
                st.subheader("🍎 Detection Results")
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = result.names[cls]
                    st.write(f"🔹 **{label}** — {conf*100:.2f}% confidence")
            else:
                st.warning("No fruits detected 😢")

# ============================================================
# 🎥 WEBCAM TAB
# ============================================================
with tab2:
    camera = st.camera_input("Capture image from webcam")

    if camera:
        img = Image.open(camera)
        img_np = np.array(img)

        st.image(img, caption="📷 Captured Image", width=400)

        with st.spinner("Detecting fruits..."):
            results = model.predict(img_np, conf=0.5)
            result = results[0]

            annotated = img_np.copy()
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                text = f"{label} ({conf*100:.1f}%)"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (x1, y1 - 25), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(annotated, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated, caption="✅ Detected Fruits", width=550)

            if len(result.boxes) > 0:
                st.subheader("🍇 Detection Results")
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = result.names[cls]
                    st.write(f"🔸 **{label}** — {conf*100:.2f}% confidence")
            else:
                st.warning("No fruits detected 😢")

# ============================================================
# ✅ END OF APP
# ============================================================
