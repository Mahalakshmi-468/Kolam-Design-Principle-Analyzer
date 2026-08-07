import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
import random

from image_processing import (
    convert_to_gray,
    remove_noise,
    detect_edges,
    detect_contours,
    detect_kolam_region,
    detect_dots,
    compare_with_recreation
)

from feature_extraction import (
    count_contours,
    estimate_symmetry,
    estimate_full_symmetry,
    classify_pattern,
    calculate_complexity,
    estimate_grid_size,
    get_design_style,
    get_difficulty,
    get_drawing_method,
    estimate_time,
    recreation_status,
    confidence_score
)

from kolam_recreation import recreate_kolam
from kolam_generator import generate_kolam_design
from database import save_analysis, get_history, clear_history
from report_generator import generate_pdf_report

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Kolam Design Principle Analyzer",
    page_icon="🎨",
    layout="wide"
)

# ---------------------------------
# Traditional Theme (Kolam colors: maroon, gold, cream)
# ---------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Yatra+One&family=Mukta:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Mukta', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #FDF6E9 0%, #FBF0DC 100%);
}

/* Decorative traditional header banner */
.kolam-banner {
    background: linear-gradient(120deg, #7A1F2B 0%, #A32638 55%, #7A1F2B 100%);
    padding: 28px 20px 22px 20px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 6px 18px rgba(122, 31, 43, 0.35);
    margin-bottom: 6px;
    border: 2px solid #D4A017;
}

.kolam-banner h1 {
    font-family: 'Yatra One', cursive;
    color: #FDF0D5;
    font-size: 2.4rem;
    margin: 0 0 6px 0;
    letter-spacing: 1px;
}

.kolam-banner p {
    color: #F3D9A4;
    font-size: 1rem;
    margin: 0;
    letter-spacing: 0.5px;
}

.kolam-divider {
    height: 3px;
    margin: 18px 0 22px 0;
    background: repeating-linear-gradient(
        90deg,
        #D4A017 0px, #D4A017 10px,
        transparent 10px, transparent 20px
    );
    border-radius: 4px;
}

/* Section headers */
h2, h3 {
    color: #7A1F2B !important;
    font-family: 'Mukta', sans-serif;
    font-weight: 700 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFDF8;
    border: 1.5px solid #D4A017;
    border-radius: 12px;
    padding: 14px 10px;
    box-shadow: 0 2px 6px rgba(122, 31, 43, 0.08);
}

[data-testid="stMetricLabel"] {
    color: #7A1F2B !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #A32638 !important;
}

/* Bordered containers used for image cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    border: 2px solid #D4A017 !important;
    background-color: #FFFDF8 !important;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background-color: #7A1F2B;
    color: #FDF0D5;
    border-radius: 8px;
    border: 1.5px solid #D4A017;
    font-weight: 600;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: #A32638;
    border-color: #D4A017;
    color: #FFFFFF;
}

.card-title {
    text-align: center;
    color: #7A1F2B;
    font-family: 'Mukta', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Title
# ---------------------------------
st.markdown("""
<div class="kolam-banner">
<h1>🎨 Kolam Design Principle Analyzer</h1>
<p>Computer Vision Based Analysis of Traditional Tamil Kolam Designs</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="kolam-divider"></div>', unsafe_allow_html=True)

st.success("✅ Upload a Kolam image to begin the analysis.")

# ---------------------------------
# Sidebar: Analysis History
# ---------------------------------
with st.sidebar:
    st.header("📜 Analysis History")

    history_df = get_history()

    if history_df.empty:
        st.info("No past analyses yet. Upload a Kolam image to get started.")
    else:
        st.dataframe(
            history_df[[
                "timestamp", "pattern", "dot_count",
                "complexity", "confidence"
            ]],
            width='stretch',
            hide_index=True
        )

        if st.button("🗑 Clear History"):
            clear_history()
            st.rerun()

# ---------------------------------
# Generate New Kolam Design (Generative Mode)
# ---------------------------------
with st.expander("🪔 Generate a New Original Kolam Design", expanded=False):

    st.caption(
        "Procedurally create a brand-new Kolam pattern from a symmetric "
        "dot grid, instead of analyzing an uploaded photo."
    )

    gen_col1, gen_col2, gen_col3 = st.columns(3)

    with gen_col1:
        gen_grid_size = st.slider("Grid Size (dots per side)", 5, 13, 7, step=2)

    with gen_col2:
        gen_symmetry_label = st.selectbox(
            "Symmetry Mode",
            ["4-Fold Symmetric", "2-Fold (Vertical Mirror)", "2-Fold (Horizontal Mirror)"]
        )
        gen_symmetry_map = {
            "4-Fold Symmetric": "4-fold",
            "2-Fold (Vertical Mirror)": "2-fold-vertical",
            "2-Fold (Horizontal Mirror)": "2-fold-horizontal",
        }
        gen_symmetry = gen_symmetry_map[gen_symmetry_label]

    with gen_col3:
        gen_density = st.slider("Pattern Density", 0.2, 0.9, 0.55)

    if st.button("✨ Generate New Kolam Design"):
        st.session_state["gen_seed"] = random.randint(0, 999999)

    if "gen_seed" not in st.session_state:
        st.session_state["gen_seed"] = 42

    generated_canvas, gen_dot_count, gen_edge_count = generate_kolam_design(
        grid_size=gen_grid_size,
        symmetry=gen_symmetry,
        density=gen_density,
        seed=st.session_state["gen_seed"]
    )

    with st.container(border=True):
        st.markdown('<p class="card-title">🎨 Generated Kolam Design</p>', unsafe_allow_html=True)
        st.image(generated_canvas, channels="BGR", width='stretch')
        st.caption(f"{gen_dot_count} dots • {gen_edge_count} thread connections • {gen_symmetry_label}")

    success, gen_png = cv2.imencode(".png", generated_canvas)

    if success:
        st.download_button(
            label="📥 Download Generated Kolam (PNG)",
            data=gen_png.tobytes(),
            file_name="generated_kolam_design.png",
            mime="image/png"
        )

st.markdown('<div class="kolam-divider"></div>', unsafe_allow_html=True)

# ---------------------------------
# Upload / Capture Image
# ---------------------------------
st.subheader("🔍 Analyze an Existing Kolam")

input_mode = st.radio(
    "Choose Input Method",
    ["📁 Upload Image", "📷 Capture from Camera"],
    horizontal=True
)

if input_mode == "📁 Upload Image":
    uploaded_file = st.file_uploader(
        "📁 Upload a Kolam Image",
        type=["jpg", "jpeg", "png"]
    )
else:
    uploaded_file = st.camera_input("📷 Capture a Kolam using your camera")

# ---------------------------------
# Process Image
# ---------------------------------
if uploaded_file is not None:
    start_time = time.time()
    progress_bar = st.progress(0)
    status_text = st.empty()

    st.success("✅ Image uploaded successfully!")

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Unable to load image.")
        st.stop()

    # -----------------------------
    # Grayscale
    # -----------------------------
    gray_image = convert_to_gray(image)

    progress_bar.progress(10)
    status_text.text("🔍 Converting image to grayscale...")

    # -----------------------------
    # Kolam Region Detection
    # -----------------------------
    kolam_region = detect_kolam_region(gray_image)

    progress_bar.progress(20)
    status_text.text("🎯 Detecting Kolam region...")

    # -----------------------------
    # Noise Removal
    # -----------------------------
    noise_removed = remove_noise(gray_image)
    progress_bar.progress(35)
    status_text.text("🧹 Removing image noise...")

    # -----------------------------
    # Edge Detection
    # -----------------------------
    edge_image = detect_edges(noise_removed)

    progress_bar.progress(50)
    status_text.text("✂ Detecting edges...")
    # -----------------------------
    # Contour Detection
    # -----------------------------
    contours = detect_contours(edge_image)

    contour_count = count_contours(contours)

    contour_image = image.copy()
    
    cv2.drawContours(
    contour_image,
    contours,
    -1,
    (0,255,0),
    1
)


    progress_bar.progress(60)
    status_text.text("📐 Detecting contours...")
    # -----------------------------
    # Dot Detection
    # -----------------------------


    dots = detect_dots(noise_removed)
    st.write("Detected Dots:", len(dots))
    dot_image = image.copy()

    dot_count = 0
    if dots is None:
        dots = []
    if len(dots) > 0:
        dot_count = len(dots)
        count = 1

    for (x,y,r) in dots:

        cv2.circle(
            dot_image,
            (x,y),
            r,
            (0,255,0),
            2
        )

        cv2.circle(
            dot_image,
            (x,y),
            2,
            (0,0,255),
            -1
        )

        cv2.putText(
            dot_image,
            str(count),
            (x+5,y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255,0,0),
            1
        )

        count += 1

    progress_bar.progress(70)
    status_text.text("🔵 Detecting Kolam dots...")
    # -----------------------------
    # Symmetry Detection
    # -----------------------------
    symmetry_level, similarity = estimate_symmetry(gray_image)
    full_symmetry = estimate_full_symmetry(gray_image)

    progress_bar.progress(80)
    status_text.text("📏 Calculating symmetry...")
    # -----------------------------
    # Pattern Classification
    # -----------------------------
    complexity = calculate_complexity(
        dot_count,
        contour_count
    )
    
    pattern = classify_pattern(
        dot_count,
        contour_count,
        similarity,
        complexity
    )

    progress_bar.progress(90)
    status_text.text("🧠 Classifying Kolam pattern...")
    # -----------------------------
    # Complexity Score
    # -----------------------------
    

    progress_bar.progress(95)
    status_text.text("✅ Complexity Score Calculated.")

    # -----------------------------
    # Grid Size Estimation
    # -----------------------------
    grid_size = estimate_grid_size(dot_count)
    
    # ---------------------------------
    # Additional Analysis
    # ---------------------------------
    design_style = get_design_style(pattern)
    
    difficulty = get_difficulty(complexity)
    
    drawing_method = get_drawing_method(pattern)
    
    estimated_time = estimate_time(dot_count)
    
    status = recreation_status()

    confidence = confidence_score(similarity, complexity)

    progress_bar.progress(95)
    status_text.text("📊 Preparing dashboard...")
    # -----------------------------
    # Kolam Recreation
    # -----------------------------
    
    recreated_kolam = recreate_kolam(dots)

    # -----------------------------
    # Original vs Recreated Difference Heatmap
    # -----------------------------
    diff_heatmap, match_score = compare_with_recreation(image, recreated_kolam)

    progress_bar.progress(100)
    status_text.success("✅ Analysis Completed Successfully!")
    # -----------------------------
    # Display Images
    # -----------------------------
    st.subheader("🖼 Image Processing Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown('<p class="card-title">📷 Original Image</p>', unsafe_allow_html=True)
            st.image(image, channels="BGR", width='stretch')

    with col2:
        with st.container(border=True):
            st.markdown('<p class="card-title">⚫ Grayscale Image</p>', unsafe_allow_html=True)
            st.image(gray_image, width='stretch')

    with col3:
        with st.container(border=True):
            st.markdown('<p class="card-title">🧹 Noise Removed</p>', unsafe_allow_html=True)
            st.image(noise_removed, width='stretch')

    col4, col5, col6 = st.columns(3)

    with col4:
        with st.container(border=True):
            st.markdown('<p class="card-title">✂ Edge Detection</p>', unsafe_allow_html=True)
            st.image(edge_image, width='stretch')

    with col5:
        with st.container(border=True):
            st.markdown('<p class="card-title">📐 Contours</p>', unsafe_allow_html=True)
            st.image(contour_image, channels="BGR", width='stretch')

    with col6:
        with st.container(border=True):
            st.markdown('<p class="card-title">🔵 Detected Dots</p>', unsafe_allow_html=True)
            st.image(dot_image, channels="BGR", width='stretch')

    # -----------------------------
    # Final Analysis Dashboard
    # -----------------------------
    st.markdown('<div class="kolam-divider"></div>', unsafe_allow_html=True)
    st.subheader("📊 Final Kolam Analysis Dashboard")

    st.subheader("🖌 Original vs Recreated Kolam")
    st.caption(
        "Each version is shown in its own panel so you can compare "
        "the hand-drawn original against the computer-generated "
        "structural recreation side by side."
    )

    orig_col, recreated_col = st.columns(2)

    with orig_col:
        with st.container(border=True):
            st.markdown('<p class="card-title">📷 Original Kolam</p>', unsafe_allow_html=True)
            st.image(image, channels="BGR", width='stretch')

    with recreated_col:
        with st.container(border=True):
            st.markdown('<p class="card-title">🎨 AI Recreated Kolam</p>', unsafe_allow_html=True)
            st.image(recreated_kolam, channels="BGR", width='stretch')

    st.subheader("🔥 Difference Heatmap")
    st.caption(
        "Warmer colors (red/yellow) show where the recreated Kolam "
        "diverges from the original's structure; cooler/darker areas "
        "indicate a close match."
    )

    with st.container(border=True):
        st.markdown('<p class="card-title">Structural Comparison</p>', unsafe_allow_html=True)
        st.image(
            diff_heatmap,
            channels="BGR",
            caption=f"Structural Match Score: {match_score}%",
            width='stretch'
        )

    st.subheader("🌀 Full Symmetry Analysis")
    st.caption(
        "Checks reflective symmetry across both axes plus 180° "
        "rotational symmetry, and summarizes the overall symmetry type."
    )

    sym_col1, sym_col2, sym_col3, sym_col4 = st.columns(4)

    with sym_col1:
        st.metric(
            "Horizontal Axis",
            f"{full_symmetry['horizontal_similarity']}%",
            "✔ Symmetric" if full_symmetry['horizontal_symmetric'] else "✘ Not Symmetric"
        )

    with sym_col2:
        st.metric(
            "Vertical Axis",
            f"{full_symmetry['vertical_similarity']}%",
            "✔ Symmetric" if full_symmetry['vertical_symmetric'] else "✘ Not Symmetric"
        )

    with sym_col3:
        st.metric(
            "180° Rotational",
            f"{full_symmetry['rotational_similarity']}%",
            "✔ Symmetric" if full_symmetry['rotational_symmetric'] else "✘ Not Symmetric"
        )

    with sym_col4:
        st.metric("Overall Symmetry Type", full_symmetry['symmetry_type'])

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Kolam Type", pattern)

    with col2:
        st.metric("Design Style", design_style)

    with col3:
        st.metric("Grid Pattern", grid_size)
        
    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("Symmetry", symmetry_level)

    with col5:
        st.metric("Difficulty", difficulty)

    with col6:
        st.metric("Complexity", f"{complexity}/100")


    col7, col8, col9, col10 = st.columns(4)

    with col7:
        st.metric("Drawing Method", drawing_method)

    with col8:
        st.metric("Estimated Time", estimated_time)

    with col9:
        st.metric("Confidence", f"{confidence}%")

    with col10:
        st.metric("Recreation Status", status)

    col11, col12 = st.columns(2)

    with col11:
        st.metric("Structural Match Score", f"{match_score}%")

    # -----------------------------
    # Final Report
    # -----------------------------
    # ---------------------------------
    # Design Principles
    # ---------------------------------
    st.subheader("🧠 Design Principles Identified")

    principles = []

    if "Pulli" in pattern:
        principles.append("✔ Dot Matrix Construction")

    if "Sikku" in pattern:
        principles.append("✔ Continuous Loop Pattern")

    if similarity >= 85:
        principles.append("✔ Horizontal Symmetry")

    if contour_count >= 20:
        principles.append("✔ Closed Curve Structure")

    if complexity >= 70:
        principles.append("✔ High Visual Complexity")

    if dot_count >= 30:
        principles.append("✔ Medium Density Dot Grid")

    principles.append("✔ Traditional Tamil Kolam")

    for p in principles:
        st.write(p)

    # ---------------------------------
    # Final Report
    # ---------------------------------
    st.subheader("📝 Final Kolam Analysis Report")

    st.write(f"**Contours Detected:** {contour_count}")
    st.write(f"**Detected Dots:** {dot_count}")
    st.write(f"**Pattern Type:** {pattern}")
    st.write(f"**Symmetry Level:** {symmetry_level}")
    st.write(f"**Similarity Score:** {similarity:.2f}%")
    st.write(f"**Complexity Score:** {complexity}/100")
    st.write(f"**Estimated Grid Size:** {grid_size}")

    st.subheader("📄 Analysis Summary")

    summary = f"""
    The uploaded Kolam image has been successfully analyzed.

    Pattern Type        : {pattern}

    Design Style        : {design_style}

    Detected Dots       : {dot_count}

    Contours            : {contour_count}

    Grid Size           : {grid_size}

    Symmetry            : {symmetry_level}

    Similarity Score    : {similarity:.2f}%

    Complexity Score    : {complexity}/100

    Confidence Score    : {confidence}%

    Estimated Time      : {estimated_time}

    The detected design follows the traditional principles
    of Tamil Kolam art and has been successfully analyzed
    using image processing and computer vision techniques.
    """

    st.success(summary)

    st.subheader("✅ Conclusion")

    st.success(
        f"""
    The Kolam analysis has been completed successfully.

    ✔ Pattern Identified : {pattern}

    ✔ Design Style : {design_style}

    ✔ Symmetry : {symmetry_level}

    ✔ Complexity : {complexity}/100

    ✔ Confidence : {confidence}%

    ✔ Reconstruction Status : {status}

    Thank you for using the Kolam Design Principle Analyzer.
    """
    )

    # -----------------------------
    # Save this analysis to history (SQLite)
    # Uses the uploaded file's identity so each new upload gets saved
    # exactly once, even though Streamlit reruns the script on every
    # widget interaction.
    # -----------------------------
    file_signature = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("last_saved_signature") != file_signature:
        save_analysis(
            pattern=pattern,
            design_style=design_style,
            dot_count=dot_count,
            contour_count=contour_count,
            symmetry_level=symmetry_level,
            similarity=round(similarity, 2),
            complexity=complexity,
            grid_size=grid_size,
            difficulty=difficulty,
            confidence=confidence,
            match_score=match_score
        )
        st.session_state["last_saved_signature"] = file_signature

    # -----------------------------
    # Downloadable PDF Report
    # -----------------------------
    st.subheader("📥 Download Report")

    report_metrics = {
        "Pattern Type": pattern,
        "Design Style": design_style,
        "Detected Dots": dot_count,
        "Contours Detected": contour_count,
        "Grid Size": grid_size,
        "Symmetry Level": symmetry_level,
        "Similarity Score": f"{similarity:.2f}%",
        "Complexity Score": f"{complexity}/100",
        "Difficulty": difficulty,
        "Drawing Method": drawing_method,
        "Estimated Time": estimated_time,
        "Confidence Score": f"{confidence}%",
        "Structural Match Score": f"{match_score}%",
        "Recreation Status": status,
    }

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        pdf_path = tmp_pdf.name

    generate_pdf_report(
        output_path=pdf_path,
        images={
            "original": image,
            "recreated": recreated_kolam,
            "difference": diff_heatmap
        },
        metrics=report_metrics,
        principles=principles
    )

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    os.remove(pdf_path)

    st.download_button(
        label="📄 Download PDF Analysis Report",
        data=pdf_bytes,
        file_name="kolam_analysis_report.pdf",
        mime="application/pdf"
    )
