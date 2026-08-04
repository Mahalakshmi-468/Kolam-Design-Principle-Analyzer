import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd

from image_processing import (
    convert_to_gray,
    remove_noise,
    detect_edges,
    detect_contours,
    detect_kolam_region,
    detect_dots
)

from feature_extraction import (
    count_contours,
    estimate_symmetry,
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
from database import (
    create_database,
    insert_analysis,
    get_history
)

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Kolam Design Principle Analyzer",
    page_icon="🎨",
    layout="wide"
)
create_database()

# ===============================
# HEADER
# ===============================

st.markdown("""
<div style="text-align:center;">

<h1 style="color:#8B0000;">
🎨 Kolam Design Principle Analyzer
</h1>

<h4 style="color:gray;">
AI-Based Image Processing for Identifying Design Principles and
Digital Structural Reconstruction of Traditional Tamil Kolams
</h4>

</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===============================
# SIDEBAR
# ===============================

st.sidebar.title("📌 Project Information")

st.sidebar.info("""
### Developed Using

- Python
- OpenCV
- Streamlit
- NumPy
- SQLite

### Domain

Computer Vision

Image Processing

Pattern Recognition

### Department

Computer Science and Business Systems

V.S.B Engineering College
""")

st.subheader("📖 About the Project")

st.write("""
This application analyzes traditional Tamil Kolam images using
Computer Vision and Image Processing techniques.

The system identifies:

- Pattern Type
- Dot Count
- Contours
- Symmetry
- Complexity
- Grid Size
- Design Style

Finally, it recreates the Kolam digitally and stores
the analysis using SQLite.
""")

st.success("✅ Upload a Kolam image to begin the analysis.")
st.info("""
### 📌 Project Highlights

✔ Computer Vision Based Analysis

✔ Image Processing

✔ Digital Structural Reconstruction

✔ SQLite Database

✔ Traditional Tamil Kolam Recognition

✔ Streamlit Dashboard
""")

# ---------------------------------
# Upload Image
# ---------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload a Kolam Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a Pulli or Sikku Kolam."
)

# ---------------------------------
# Process Image
# ---------------------------------
if uploaded_file is not None:
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
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)

    progress_bar.progress(60)
    status_text.text("📐 Detecting contours...")

    # -----------------------------
    # Dot Detection
    # -----------------------------
    dots = detect_dots(noise_removed)
    if dots is None:
        dots = []

    dot_count = len(dots)
    dot_image = image.copy()

    for i, (x, y, r) in enumerate(dots, start=1):
        cv2.circle(dot_image, (x, y), r, (0, 255, 0), 2)
        cv2.circle(dot_image, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(
            dot_image,
            str(i),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1
        )

    progress_bar.progress(70)
    status_text.text("🔵 Detecting Kolam dots...")

    # -----------------------------
    # Symmetry Detection
    # -----------------------------
    symmetry_level, similarity = estimate_symmetry(gray_image)
    progress_bar.progress(80)
    status_text.text("📏 Calculating symmetry...")

    # -----------------------------
    # Complexity + Pattern Classification
    # -----------------------------
    complexity = calculate_complexity(dot_count, contour_count)
    pattern = classify_pattern(dot_count, contour_count, similarity, complexity)

    progress_bar.progress(90)
    status_text.text("🧠 Classifying Kolam pattern...")

    # -----------------------------
    # Grid Size + Additional Analysis
    # -----------------------------
    grid_size = estimate_grid_size(dot_count)
    design_style = get_design_style(pattern)
    difficulty = get_difficulty(complexity)
    drawing_method = get_drawing_method(pattern)
    estimated_time = estimate_time(dot_count)
    status = recreation_status()
    confidence = confidence_score(similarity, complexity)

    insert_analysis(
        uploaded_file.name,
        pattern,
        dot_count,
        contour_count,
        symmetry_level,
        complexity,
        confidence
    )

    progress_bar.progress(95)
    status_text.text("📊 Preparing dashboard...")

    # -----------------------------
    # Kolam Recreation
    # -----------------------------
    recreated_kolam = recreate_kolam(image, dots)

    progress_bar.progress(100)
    status_text.success("✅ Analysis Completed Successfully!")

    # -----------------------------
    # Display Images
    # -----------------------------
    st.subheader("🖼 Image Processing Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, channels="BGR", caption="Original Image", use_container_width=True)
    with col2:
        st.image(gray_image, caption="Grayscale Image", use_container_width=True)
    with col3:
        st.image(noise_removed, caption="Noise Removed", use_container_width=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.image(edge_image, caption="Edge Detection", use_container_width=True)
    with col5:
        st.image(contour_image, channels="BGR", caption="Contours", use_container_width=True)
    with col6:
        st.image(dot_image, channels="BGR", caption="Detected Dots", use_container_width=True)

    # -----------------------------
    # Final Analysis Dashboard
    # -----------------------------
    st.subheader("📊 Final Kolam Analysis Dashboard")
    st.subheader("🎨 Digital Structural Reconstruction")
    st.image(
        recreated_kolam,
        channels="BGR",
        caption="Digital Structural Reconstruction",
        use_container_width=True
    )

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
    st.subheader("⚙ Processing Workflow")
    st.markdown("""
    1. Upload Image
    2. Convert to Grayscale
    3. Remove Noise
    4. Detect Edges
    5. Detect Contours
    6. Detect Dots
    7. Feature Extraction
    8. Pattern Classification
    9. Digital Structural Reconstruction
    10. Store Results in SQLite
    """)

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

    st.subheader("📈 Project Statistics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detected Dots", dot_count)
    c2.metric("Contours", contour_count)
    c3.metric("Similarity", f"{similarity:.2f}%")
    c4.metric("Confidence", f"{confidence}%")

    st.subheader("📊 Analysis Visualization")

    chart_data = pd.DataFrame({
        "Metric": ["Dots", "Contours", "Similarity", "Confidence"],
        "Value": [dot_count, contour_count, similarity, confidence]
    })

    st.bar_chart(chart_data.set_index("Metric"))

    st.subheader("✅ Conclusion")

    st.success(f"""
    The Kolam analysis has been completed successfully.

    ✔ Pattern Identified : {pattern}

    ✔ Design Style : {design_style}

    ✔ Symmetry : {symmetry_level}

    ✔ Complexity : {complexity}/100

    ✔ Confidence : {confidence}%

    ✔ Reconstruction Status : {status}

    Thank you for using the Kolam Design Principle Analyzer.
    """)

    st.subheader("🛠 Technologies Used")

    tech_df = pd.DataFrame({
        "Technology": ["Python", "OpenCV", "NumPy", "Streamlit", "SQLite"],
        "Purpose": [
            "Programming",
            "Image Processing",
            "Scientific Computing",
            "User Interface",
            "Database"
        ]
    })

    st.table(tech_df)

    # ---------------------------------
    # Previous Analysis History
    # ---------------------------------
    st.subheader("📂 Previous Analysis")

    history = get_history()

    if len(history) > 0:
        history_df = pd.DataFrame(
            history,
            columns=[
                "ID",
                "Image",
                "Pattern",
                "Dots",
                "Contours",
                "Symmetry",
                "Complexity",
                "Confidence",
                "Date"
            ]
        )
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No previous analysis found.")

else:
    st.info("📂 Please upload a Kolam image above to begin the analysis.")
