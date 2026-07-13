import streamlit as st
import cv2
import numpy as np
import time

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

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Kolam Design Principle Analyzer",
    page_icon="🎨",
    layout="wide"
)

# ---------------------------------
# Title
# ---------------------------------
st.markdown("""
# 🎨 Kolam Design Principle Analyzer

### Computer Vision Based Analysis of Traditional Tamil Kolam Designs

---
""")

st.success("✅ Upload a Kolam image to begin the analysis.")

# ---------------------------------
# Upload Image
# ---------------------------------
uploaded_file = st.file_uploader(
    "📁 Upload a Kolam Image",
    type=["jpg", "jpeg", "png"]
)

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
    progress_bar.progress(100)
    status_text.success("✅ Analysis Completed Successfully!")
    # -----------------------------
    # Display Images
    # -----------------------------
    st.subheader("🖼 Image Processing Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            image,
            channels="BGR",
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            gray_image,
            caption="Grayscale Image",
            use_container_width=True
        )

    with col3:
        st.image(
            noise_removed,
            caption="Noise Removed",
            use_container_width=True
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.image(
            edge_image,
            caption="Edge Detection",
            use_container_width=True
        )

    with col5:
        st.image(
            contour_image,
            channels="BGR",
            caption="Contours",
            use_container_width=True
        )

    with col6:
        st.image(
            dot_image,
            channels="BGR",
            caption="Detected Dots",
            use_container_width=True
        )

    # -----------------------------
    # Final Analysis Dashboard
    # -----------------------------
    # ---------------------------------
    # Additional Analysis
    #---------------------------------
    st.subheader("📊 Final Kolam Analysis Dashboard")
    st.subheader("🎨 Recreated Kolam")
    
    st.image(
        recreated_kolam,
        channels="BGR",
        caption="Recreated Kolam",
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