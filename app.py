"""
YOLOv8 PPE Detection System
============================
A Streamlit web application for real-time Personal Protective Equipment (PPE)
detection using a trained YOLOv8 model.

Detects: Helmet, No-Helmet, Vest, No-Vest, Person (depending on your model's classes)
Flags any class starting with "NO-" / "no-" as a safety violation.

Run with:
    pip install streamlit ultralytics opencv-python-headless pillow numpy
    streamlit run app.py
"""

import os
import io
import time
import tempfile
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Ultralytics YOLO import is wrapped so the app can still show a friendly
# error message in the UI if the package isn't installed.
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


# --------------------------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit call)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="YOLOv8 PPE Detection System",
    page_icon="🦺",
    layout="wide",
)

# Default path to the model file. Change this if your .pt file lives
# somewhere else, or just use the "Custom model path" field in the sidebar.
DEFAULT_MODEL_PATH = "best_yolov8_ppe.pt"


# --------------------------------------------------------------------------
# MODEL LOADING (cached so it only loads once per session/model path)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """
    Load a YOLOv8 model from disk. Cached with @st.cache_resource so the
    (potentially large) model weights are not reloaded on every rerun/
    interaction -- only when the model_path argument actually changes.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = YOLO(model_path)
    return model


# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------
def is_violation_class(class_name: str) -> bool:
    """Any class name starting with 'no-' (case-insensitive) is a violation."""
    return class_name.strip().lower().startswith("no-") or class_name.strip().lower().startswith("no_")


def run_inference(model, image_bgr: np.ndarray, conf: float, iou: float):
    """
    Run YOLOv8 inference on a single BGR image (OpenCV format).
    Returns:
        annotated_bgr: image with boxes/labels drawn
        detections: list of dicts {class_name, confidence}
    """
    results = model.predict(source=image_bgr, conf=conf, iou=iou, verbose=False)
    result = results[0]

    annotated_bgr = result.plot()  # ultralytics draws boxes+labels+scores for us

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            detections.append({"class_name": class_name, "confidence": confidence})

    return annotated_bgr, detections


def summarize_detections(detections):
    """Build a 'X helmets, Y no-vest' style breakdown string + counts dict."""
    counts = {}
    for det in detections:
        name = det["class_name"]
        counts[name] = counts.get(name, 0) + 1

    if not counts:
        return "No objects detected", counts

    parts = [f"{count} {name}" for name, count in counts.items()]
    return ", ".join(parts), counts


def log_violations(detections, source_label="Image"):
    """Append any violation detections to the session-state violation log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for det in detections:
        if is_violation_class(det["class_name"]):
            st.session_state.violation_log.append({
                "Timestamp": timestamp,
                "Source": source_label,
                "Class": det["class_name"],
                "Confidence": f"{det['confidence']:.2f}",
            })


def any_violation(detections) -> bool:
    return any(is_violation_class(d["class_name"]) for d in detections)


def render_violation_banner(detections):
    """Show the red/green status banner based on current detections."""
    if any_violation(detections):
        st.error("⚠️ PPE VIOLATION DETECTED", icon="⚠️")
    else:
        st.success("✅ No Violations", icon="✅")


def render_detection_summary(detections):
    summary_text, counts = summarize_detections(detections)
    st.markdown(f"**Detections:** {summary_text}")


def render_violation_log():
    with st.expander(f"📋 Violation Log ({len(st.session_state.violation_log)} entries)", expanded=False):
        if st.session_state.violation_log:
            st.dataframe(
                st.session_state.violation_log[::-1],  # newest first
                use_container_width=True,
                hide_index=True,
            )
            if st.button("Clear Violation Log"):
                st.session_state.violation_log = []
                st.rerun()
        else:
            st.write("No violations logged yet.")


# --------------------------------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------------------------------
if "violation_log" not in st.session_state:
    st.session_state.violation_log = []


# --------------------------------------------------------------------------
# SIDEBAR - CONTROLS
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

# --- Model path ---
st.sidebar.subheader("Model")
custom_path = st.sidebar.text_input(
    "Model path (.pt file)",
    value=DEFAULT_MODEL_PATH,
    help="Path to your trained YOLOv8 PPE detection weights file.",
)

model = None
model_status_placeholder = st.sidebar.empty()

if not ULTRALYTICS_AVAILABLE:
    model_status_placeholder.error(
        "❌ The 'ultralytics' package is not installed. "
        "Run: pip install ultralytics"
    )
else:
    try:
        with st.spinner("Loading model..."):
            model = load_model(custom_path)
        model_status_placeholder.success(f"✅ Model loaded: {os.path.basename(custom_path)}")
    except FileNotFoundError as e:
        model_status_placeholder.error(f"❌ {e}")
    except Exception as e:
        model_status_placeholder.error(f"❌ Failed to load model: {e}")

# --- Input mode ---
st.sidebar.subheader("Input Source")
input_mode = st.sidebar.radio(
    "Choose input mode",
    ["Image Upload", "Video Upload", "Live Webcam"],
)

# --- Detection thresholds ---
st.sidebar.subheader("Detection Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.01, max_value=0.99, value=0.50, step=0.01,
    help="Minimum confidence score for a detection to be shown.",
)
iou_threshold = st.sidebar.slider(
    "IoU Threshold", min_value=0.10, max_value=0.90, value=0.40, step=0.01,
    help="Lower values reduce overlapping/conflicting boxes (e.g. both "
         "'Vest' and 'No-Vest' on the same person).",
)

st.sidebar.markdown("---")
st.sidebar.caption("Flags any class starting with 'NO-' as a PPE violation.")


# --------------------------------------------------------------------------
# MAIN AREA - HEADER
# --------------------------------------------------------------------------
st.title("🦺 YOLOv8 PPE Detection System")
st.caption("Real-time Personal Protective Equipment compliance monitoring")
st.markdown("---")

if model is None:
    st.warning("⏳ Waiting for a valid model to be loaded. Check the sidebar for details.")
    st.stop()


# --------------------------------------------------------------------------
# MODE 1: IMAGE UPLOAD
# --------------------------------------------------------------------------
if input_mode == "Image Upload":
    st.subheader("📷 Image Detection")
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        try:
            image = Image.open(uploaded_file).convert("RGB")
            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            with col1:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)

            if st.button("🔍 Run Detection", type="primary"):
                with st.spinner("Running detection..."):
                    try:
                        annotated_bgr, detections = run_inference(
                            model, image_bgr, conf_threshold, iou_threshold
                        )
                        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                        with col2:
                            st.markdown("**Detection Result**")
                            st.image(annotated_rgb, use_container_width=True)

                        render_violation_banner(detections)
                        render_detection_summary(detections)
                        log_violations(detections, source_label=uploaded_file.name)

                    except Exception as e:
                        st.error(f"❌ Detection failed: {e}")

        except Exception as e:
            st.error(f"❌ Could not read the uploaded image: {e}")
    else:
        st.info("Upload an image to begin detection.")

    render_violation_log()


# --------------------------------------------------------------------------
# MODE 2: VIDEO UPLOAD
# --------------------------------------------------------------------------
elif input_mode == "Video Upload":
    st.subheader("🎥 Video Detection")
    uploaded_video = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        # Save to a temp file so OpenCV's VideoCapture can read it
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1])
        tfile.write(uploaded_video.read())
        tfile.close()

        start_button = st.button("🔍 Run Detection", type="primary")
        frame_display = st.empty()
        status_display = st.empty()
        summary_display = st.empty()
        progress_bar = st.progress(0)

        if start_button:
            try:
                cap = cv2.VideoCapture(tfile.name)
                if not cap.isOpened():
                    st.error("❌ Could not open the uploaded video file.")
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                    frame_idx = 0

                    # Process every Nth frame to keep things responsive for long videos
                    frame_skip = max(1, total_frames // 300)  # cap ~300 processed frames

                    with st.spinner("Processing video frame by frame..."):
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            frame_idx += 1
                            if frame_idx % frame_skip != 0:
                                continue

                            annotated_bgr, detections = run_inference(
                                model, frame, conf_threshold, iou_threshold
                            )
                            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                            frame_display.image(annotated_rgb, use_container_width=True)
                            with status_display.container():
                                render_violation_banner(detections)
                            with summary_display.container():
                                render_detection_summary(detections)

                            log_violations(detections, source_label=uploaded_video.name)

                            progress_bar.progress(min(frame_idx / total_frames, 1.0))

                    cap.release()
                    st.success("✅ Video processing complete.")

            except Exception as e:
                st.error(f"❌ Video processing failed: {e}")
            finally:
                try:
                    os.unlink(tfile.name)
                except OSError:
                    pass
    else:
        st.info("Upload a video to begin detection.")

    render_violation_log()


# --------------------------------------------------------------------------
# MODE 3: LIVE WEBCAM (snapshot-based via st.camera_input)
# --------------------------------------------------------------------------
elif input_mode == "Live Webcam":
    st.subheader("📸 Webcam Detection")
    st.caption(
        "Snapshot-based detection using your browser's camera. "
        "For continuous real-time streaming, install `streamlit-webrtc` "
        "(see note at the bottom of this section)."
    )

    camera_image = st.camera_input("Take a snapshot")

    if camera_image is not None:
        try:
            image = Image.open(camera_image).convert("RGB")
            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            with st.spinner("Running detection..."):
                annotated_bgr, detections = run_inference(
                    model, image_bgr, conf_threshold, iou_threshold
                )
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            st.image(annotated_rgb, use_container_width=True, caption="Detection Result")
            render_violation_banner(detections)
            render_detection_summary(detections)
            log_violations(detections, source_label="Webcam")

        except Exception as e:
            st.error(f"❌ Could not access or process the camera image: {e}")
    else:
        st.info("Allow camera access and take a snapshot to run detection.")

    st.markdown("---")
    with st.expander("ℹ️ Want continuous real-time streaming instead?"):
        st.markdown(
            """
For frame-by-frame *continuous* webcam streaming (rather than snapshots),
install the `streamlit-webrtc` package:

```bash
pip install streamlit-webrtc av
```

Then replace the `st.camera_input` block above with a `webrtc_streamer`
component that applies `run_inference()` inside a custom
`VideoProcessorBase.recv()` callback. This requires a running Streamlit
server reachable by your browser (works locally and on most deployments,
though corporate networks may need TURN server configuration).
            """
        )

    render_violation_log()