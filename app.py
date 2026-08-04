

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src import classification, clip_module, detection, opencv_ops, pipeline, segmentation

ROOT = Path(__file__).parent
SAMPLES = sorted((ROOT / "data" / "samples").glob("*.*"))

MODULES = {
    "OpenCV Processing": opencv_ops,
    "Object Detection": detection,
    "Classification": classification,
    "Segmentation": segmentation,
    "CLIP Zero-Shot": clip_module,
}

st.set_page_config(page_title="CV Lab", layout="wide")
st.title("Computer Vision Lab")
st.caption("Classical OpenCV, a fine-tuned classifier, YOLO detection, SAM segmentation "
           "and CLIP zero-shot tagging — all behind one interface.")


#input 

def pick_image():
    """Sample images or an upload. Samples make the app demoable in one click."""
    source = st.sidebar.radio("Image", ["Sample", "Upload"], horizontal=True)
    if source == "Sample" and SAMPLES:
        name = st.sidebar.selectbox("Sample image", [p.name for p in SAMPLES])
        return np.array(Image.open(ROOT / "data" / "samples" / name).convert("RGB")), name
    uploaded = st.sidebar.file_uploader("Upload", type=["jpg", "jpeg", "png"])
    if uploaded:
        return np.array(Image.open(uploaded).convert("RGB")), uploaded.name
    return None, None


# task controls

def opencv_controls():
    # options come straight from the dispatch table, so the two can never drift
    op = st.sidebar.selectbox("Operation", list(opencv_ops.OPERATIONS))
    cfg = {"operation": op}
    if op in ("blur", "box_blur"):
        cfg["ksize"] = st.sidebar.slider("Kernel size", 3, 31, 15 if op == "blur" else 5, step=2)
    elif op == "sharpen":
        cfg["strength"] = st.sidebar.slider("Centre weight", 1.0, 12.0, 5.0, 0.5)
    elif op == "sobel":
        cfg["direction"] = st.sidebar.radio("Edges", ["vertical", "horizontal"], horizontal=True)
    elif op == "threshold_fixed":
        cfg["thresh"] = st.sidebar.slider("Threshold", 0, 255, 127)
    elif op == "threshold_adaptive":
        cfg["block_size"] = st.sidebar.slider("Block size", 3, 51, 11, step=2)
        cfg["C"] = st.sidebar.slider("C", -20, 20, 2)
    elif op == "canny":
        cfg["low"] = st.sidebar.slider("Low threshold", 0, 255, 50)
        cfg["high"] = st.sidebar.slider("High threshold", 0, 255, 150)
        cfg["pre_blur"] = st.sidebar.checkbox("Blur first", True)
    elif op == "contours":
        cfg["min_area"] = st.sidebar.slider("Min area (px)", 0, 5000, 500, step=50)
    return cfg


def task_controls(task, image):
    if task == "OpenCV Processing":
        return opencv_controls()

    if task == "Object Detection":
        return {"conf": st.sidebar.slider("Confidence", 0.0, 1.0, 0.25),
                "iou": st.sidebar.slider("IoU (NMS)", 0.0, 1.0, 0.45)}

    if task == "CLIP Zero-Shot":
        raw = st.sidebar.text_area("Labels (one per line)",
                                   "a forest\na city\na river\nfarmland\na mountain range")
        return {"labels": [l.strip() for l in raw.splitlines() if l.strip()]}

    if task == "Segmentation":
        h, w = image.shape[:2]
        kind = st.sidebar.radio("Prompt with", ["Box", "Point"], horizontal=True)
        if kind == "Box":
            st.sidebar.caption("SAM segments inside the box. A tight box is what the "
                               "pipeline feeds it from YOLO.")
            x1, x2 = st.sidebar.slider("x range", 0, w, (w // 4, 3 * w // 4))
            y1, y2 = st.sidebar.slider("y range", 0, h, (h // 4, 3 * h // 4))
            return {"box": [x1, y1, x2, y2]}
        st.sidebar.caption("A point is ambiguous — land it on background and you get "
                           "the background.")
        return {"point": [st.sidebar.slider("x", 0, w, w // 2),
                          st.sidebar.slider("y", 0, h, h // 2)]}

    return {}


#  display

def show_scores(rows):
    
    df = pd.DataFrame([{"label": r["label"], "score": r["score"]} for r in rows])
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={"score": st.column_config.ProgressColumn(
            "score", min_value=0.0, max_value=1.0, format="%.4f")},
    )


def show_result(task, result, image):
    left, right = st.columns(2)
    left.image(image, caption="Input", width="stretch")
    if result.overlay is not None:
        right.image(result.overlay, caption=f"{task} — output", width="stretch")
    else:
        right.info("returning structured data only because machine understands data only")

    data = result.data
    if "predictions" in data:
        st.subheader(f"Top-1: {data['top1']}")
        show_scores(data["predictions"])
    elif "scores" in data:
        st.subheader(f"Top tag: {data['top']}")
        show_scores(data["scores"])
    elif "objects" in data:
        st.subheader(f"{data['count']} object(s)")
        if data["objects"]:
            show_scores(data["objects"])
    elif "mask_area_px" in data:
        a, b, c = st.columns(3)
        a.metric("Mask area", f"{data['mask_area_px']:,} px")
        b.metric("Image covered", f"{data['mask_fraction'] * 100:.1f}%")
        c.metric("SAM score", f"{data['mask_score']:.3f}")

    with st.expander("Raw output (data)"):
        st.json(data)
    with st.expander("meta — model, settings, runtime"):
        st.json(result.meta)


#app 

mode = st.sidebar.radio("Mode", ["Single Task", "Full Pipeline"])
image, image_name = pick_image()

if image is None:
    st.info("Pick a sample image or upload one to begin.")
    st.stop()

st.sidebar.divider()

if mode == "Single Task":
    task = st.sidebar.selectbox("Task", list(MODULES))
    config = task_controls(task, image)


    if task == "OpenCV Processing":
        show_result(task, opencv_ops.run(image, config), image)
    else:
        if st.sidebar.button("Run", type="primary", width="stretch"):
            with st.spinner(f"Running {task}… first call loads the model."):
                st.session_state["single"] = (task, MODULES[task].run(image, config))
        cached = st.session_state.get("single")
        if cached and cached[0] == task:
            show_result(task, cached[1], image)
        else:
            st.info("Set your options, then press **Run**.")

else:
    st.sidebar.caption("Detection → classification → SAM on the best box → CLIP tags.")
    if st.sidebar.button("Run pipeline", type="primary", width="stretch"):
        with st.spinner("Running all stages… first call loads three models."):
            st.session_state["pipeline"] = pipeline.run(image, {})

    result = st.session_state.get("pipeline")
    if result is None:
        st.info("Press **Run pipeline** to chain every module into one scene report.")
    else:
        report = result.data
        left, right = st.columns(2)
        left.image(image, caption="Input", width="stretch")
        if result.overlay is not None:
            right.image(result.overlay, caption="Best detection, segmented",
                        width="stretch")

        scene = report["scene"]
        tags = scene.get("clip_tags") or [{"label": "—"}]
        a, b, c = st.columns(3)
        a.metric("Objects found", len(report["objects"]))
        b.metric("Top CLIP tag", tags[0]["label"])
        c.metric("Total runtime", f"{result.meta['runtime_s']}s")

        st.subheader("Stage runtimes")
        st.bar_chart(pd.Series(result.meta["stages"], name="seconds"), horizontal=True)

        st.subheader("scene_report.json")
        st.caption("The overlays are for humans. This is the deliverable.")
        st.json(report)
        st.download_button(
            "Download scene_report.json",
            json.dumps(report, indent=2),
            file_name=f"scene_report_{Path(image_name).stem}.json",
            mime="application/json",
        )
