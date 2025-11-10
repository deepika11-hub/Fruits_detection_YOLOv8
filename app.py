# ============================================================
# 🍎 Streamlit Fruit Detection App (Bujji Edition 💖)
# ============================================================

import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os
import base64

# ✅ Disable GUI functions (prevents libGL errors on Streamlit Cloud)
cv2.imshow = lambda *args, **kwargs: None
cv2.waitKey = lambda *args, **kwargs: None
cv2.destroyAllWindows = lambda *args, **kwargs: None

# ============================================================
# 🌈 PAGE SETUP
# ============================================================
st.set_page_config(page_title="🍉Fruit Detection", page_icon="🍎", layout="centered")

# ============================================================
# 🍋 BACKGROUND SETUP
# ============================================================
def add_bg_image(image_path):
    """Set custom background image."""
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
            </style>
            """,
            unsafe_allow_html=True
        )

# 👇 your background file path
add_bg_image("fruit_bg.jpg")

# ============================================================
# 🍎 LOAD YOLO MODEL
# ============================================================
model_path = "best.pt"  # your trained model file
if not os.path.exists(model_path):
    st.error("❌ best.pt not found! Please place it in the same folder as app.py.")
    st.stop()
else:
    model = YOLO(model_path)

# ============================================================
# 🍊 TITLE
# ============================================================
st.title("🍓Fruit Detection With YOLOv8🍊")
st.write("Upload or capture a fruit image — view bounding boxes, labels & accuracy! 🍇")

# ============================================================
# 📸 TWO TABS: UPLOAD & CAMERA
# ============================================================
tab1, tab2 = st.tabs(["📁 **Upload Fruit Image**", "🎥 **Detect with Webcam**"])

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
            # Run YOLO prediction
            results = model.predict(img_np, conf=0.5)
            result = results[0]

            # Annotate manually for precise control
            annotated = img_np.copy()
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                text = f"{label} ({conf*100:.1f}%)"

                # Draw bounding box + label
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (x1, y1 - 25), (x1 + w, y1), (255, 0, 0), -1)
                cv2.putText(annotated, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated, caption="✅ Detected Fruits", width=550)

            # Show result text
            if len(result.boxes) > 0:
                st.subheader("🍎 Detection Results")
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = result.names[cls]
                    st.write(f"🔹 **{label}** — {conf*100:.2f}% confidence")
            else:
                st.warning("😢 No fruits detected!")

# ============================================================
# 🎥 CAMERA TAB
# ============================================================
with tab2:
    camera = st.camera_input("📷 Take a picture")

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
                st.warning("😢 No fruits detected!")

# ============================================================
# ✅ END OF APP
# ============================================================
