import streamlit as st
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import os
import time

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SambaGuard AI",
    page_icon="🌽",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────
CLASS_NAMES = [
    "fall armyworm egg",
    "fall armyworm frass",
    "fall armyworm larva",
    "fall armyworm larval damage",
]

CLASS_COLORS = {
    "fall armyworm egg"          : (255, 220, 0),
    "fall armyworm frass"        : (255, 140, 0),
    "fall armyworm larva"        : (220, 50, 50),
    "fall armyworm larval damage": (50, 100, 220),
}

INPUT_SIZE    = 640
IOU_THRESHOLD = 0.45
HF_REPO_ID    = "ndunge23/SambaGuard-v2"
MODEL_FILE    = "weights/best.onnx"

# ── Load model (cached so it only downloads once) ─────────────
@st.cache_resource
def load_model():
    with st.spinner("Loading SambaGuard model from Hugging Face..."):
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)
        session = ort.InferenceSession(model_path)
    return session

# ── Inference ─────────────────────────────────────────────────
def parse_detections(output, orig_h, orig_w, conf_threshold):
    predictions = output[0].squeeze(0).T
    boxes        = predictions[:, :4]
    class_scores = predictions[:, 4:]
    confidences  = np.max(class_scores, axis=1)
    class_ids    = np.argmax(class_scores, axis=1)

    mask        = confidences >= conf_threshold
    boxes       = boxes[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]

    if len(boxes) == 0:
        return []

    scale_x = orig_w / INPUT_SIZE
    scale_y = orig_h / INPUT_SIZE

    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * scale_x
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * scale_y
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * scale_x
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * scale_y

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        confidences.tolist(),
        conf_threshold,
        IOU_THRESHOLD,
    )

    detections = []
    for i in indices:
        detections.append({
            "class_name" : CLASS_NAMES[class_ids[i]],
            "confidence" : float(confidences[i]),
            "box"        : [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
        })
    return detections


def draw_detections(image_np, detections):
    img = image_np.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label  = det["class_name"]
        conf   = det["confidence"]
        color  = CLASS_COLORS.get(label, (0, 255, 0))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def run_inference(session, image_pil, conf_threshold):
    img_np     = np.array(image_pil.convert("RGB"))
    orig_h, orig_w = img_np.shape[:2]

    img_resized = cv2.resize(img_np, (INPUT_SIZE, INPUT_SIZE))
    img_input   = img_resized.astype(np.float32) / 255.0
    img_input   = np.transpose(img_input, (2, 0, 1))
    img_input   = np.expand_dims(img_input, axis=0)

    input_name = session.get_inputs()[0].name

    start   = time.time()
    outputs = session.run(None, {input_name: img_input})
    elapsed = time.time() - start

    detections = parse_detections(outputs, orig_h, orig_w, conf_threshold)
    result_img = draw_detections(img_np, detections)

    return result_img, detections, elapsed


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("SambaGuard AI")
    st.caption("Edge AI for Fall Armyworm Detection")
    st.divider()

    st.subheader("Model")
    st.markdown("""
- **Architecture:** YOLOv8s
- **Input size:** 640 × 640
- **Export format:** ONNX
- **Training dataset:** KaraAgro AI Maize
    """)

    st.divider()
    st.subheader("Classes")
    for name, color in CLASS_COLORS.items():
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)
        st.markdown(
            f'<span style="background:{hex_color};padding:2px 8px;'
            f'border-radius:4px;color:white;font-size:13px;">{name}</span>',
            unsafe_allow_html=True,
        )
        st.write("")

    st.divider()
    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Detections below this confidence score are ignored."
    )

    st.divider()
    st.markdown(
        "[Model on Hugging Face](https://huggingface.co/ndunge23/SambaGuard-v2/tree/main)  \n"
        "[GitHub Repository](https://github.com/aneneahs-kanaks/SambaGuard)"
    )

# ── Main ──────────────────────────────────────────────────────
st.title("SambaGuard AI — Fall Armyworm Detector")
st.markdown(
    "Upload a maize field image or use your camera to detect "
    "Fall Armyworm eggs, frass, larvae, and larval damage in real time."
)

# load model
session = load_model()

# ── Tabs ──────────────────────────────────────────────────────
tab_upload, tab_camera = st.tabs(["Upload Image", "Camera Capture"])

# ── Tab 1: Upload ─────────────────────────────────────────────
with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose a maize field image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original image")
            st.image(image, use_container_width=True)

        with st.spinner("Running detection..."):
            result_img, detections, elapsed = run_inference(
                session, image, conf_threshold
            )

        with col2:
            st.subheader("Detection result")
            st.image(result_img, use_container_width=True)

        # Results summary
        st.divider()
        st.subheader("Detection summary")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Objects detected", len(detections))
        col_b.metric("Inference time", f"{elapsed*1000:.0f} ms")
        col_c.metric("Confidence threshold", f"{conf_threshold:.0%}")

        if detections:
            st.markdown("**Detected objects:**")
            for det in detections:
                color = CLASS_COLORS.get(det["class_name"], (0,255,0))
                hex_color = "#{:02x}{:02x}{:02x}".format(*color)
                st.markdown(
                    f'<span style="background:{hex_color};padding:2px 8px;'
                    f'border-radius:4px;color:white;font-size:13px;">'
                    f'{det["class_name"]}</span> '
                    f'— confidence: **{det["confidence"]:.3f}**',
                    unsafe_allow_html=True,
                )
                st.write("")
        else:
            st.info("No FAW detected at this confidence threshold. "
                    "Try lowering the threshold in the sidebar.")

# ── Tab 2: Camera ─────────────────────────────────────────────
with tab_camera:
    st.markdown(
        "Point your camera at a maize plant and take a photo. "
        "The model will detect any Fall Armyworm signs present."
    )

    camera_image = st.camera_input("Take a photo")

    if camera_image is not None:
        image = Image.open(camera_image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Captured image")
            st.image(image, use_container_width=True)

        with st.spinner("Running detection..."):
            result_img, detections, elapsed = run_inference(
                session, image, conf_threshold
            )

        with col2:
            st.subheader("Detection result")
            st.image(result_img, use_container_width=True)

        # Results summary
        st.divider()
        st.subheader("Detection summary")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Objects detected", len(detections))
        col_b.metric("Inference time", f"{elapsed*1000:.0f} ms")
        col_c.metric("Confidence threshold", f"{conf_threshold:.0%}")

        if detections:
            st.markdown("**Detected objects:**")
            for det in detections:
                color = CLASS_COLORS.get(det["class_name"], (0,255,0))
                hex_color = "#{:02x}{:02x}{:02x}".format(*color)
                st.markdown(
                    f'<span style="background:{hex_color};padding:2px 8px;'
                    f'border-radius:4px;color:white;font-size:13px;">'
                    f'{det["class_name"]}</span> '
                    f'— confidence: **{det["confidence"]:.3f}**',
                    unsafe_allow_html=True,
                )
                st.write("")
        else:
            st.info("No FAW detected at this confidence threshold. "
                    "Try lowering the threshold in the sidebar.")
