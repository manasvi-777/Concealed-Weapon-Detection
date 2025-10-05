import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import cv2
import os

# Load YOLOv8 model once and cache it
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Make sure best.pt is in the same folder

model = load_model()

st.title("Concealed Weapon Detection (YOLOv8)")
st.write("Upload an **image** or **video** to detect concealed weapons.")

input_type = st.radio("Select input type:", ("Image", "Video"))

if input_type == "Image":
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Running detection...")
        results = model.predict(np.array(image))
        result_img = results[0].plot()
        st.image(result_img, caption="Detection Result", use_column_width=True)
        st.write("Detected objects:")
        for box in results[0].boxes:
            cls = results[0].names[int(box.cls[0])]
            conf = float(box.conf[0])
            st.write(f"- {cls} ({conf:.2f})")

else:
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_vid:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_vid.read())
        tfile.close()
        st.video(tfile.name)

        # Process video frame by frame
        st.write("Running detection on video (may take a while)...")
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        # Output video (use local directory path for Windows)
        out_path = os.path.join(os.getcwd(), "processed_video_out.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame)
            result_frame = results[0].plot()
            out.write(result_frame)
            count += 1
            progress.progress(min(count / frame_count, 1.0))
        cap.release()
        out.release()

        st.success("Detection complete! Below is the processed video:")

        # Check if video file exists and display it
        if os.path.exists(out_path):
            st.video(out_path)
        else:
            st.error(f"Video file not found at {out_path}")

        # Add download button for the processed video
        with open(out_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name=os.path.basename(out_path),
                mime="video/mp4"
            )

        os.remove(tfile.name)  # Clean up temp file
