# 🎨 Kolam Design Principle Analyzer

A computer-vision based web app for analyzing, recreating, and generating traditional **Tamil Kolam** designs. Upload a photo of a Kolam and the app will detect its dots and contours, classify its pattern type, score its symmetry and complexity, recreate it algorithmically, and export everything as a PDF report — or skip analysis entirely and generate a brand-new, original Kolam from a symmetric dot grid.

Built with **Streamlit** and **OpenCV**.

---

## ✨ Features

### 🔍 Kolam Analysis
- **Image preprocessing** — grayscale conversion, noise removal (Gaussian blur), and edge detection (Canny).
- **Kolam region detection** — isolates the design from its background using adaptive (Otsu) thresholding.
- **Dot detection** — locates the dot grid (Pulli) a Kolam is built around; for continuous-thread (Sikku/Kambi) designs with no visible dots, it falls back to detecting the enclosed loop "eyes" of the thread and uses those as structural points.
- **Contour detection** — counts and overlays the design's line/curve contours.
- **Symmetry analysis**
  - Basic horizontal similarity score with a 3-tier rating (Highly / Moderately / Low Symmetric).
  - Full symmetry breakdown across horizontal, vertical, and 180° rotational axes, summarized as 4-Fold, Bi-Axial, Single-Axis, or Asymmetric.
- **Pattern classification** — categorizes the design as Sikku Kolam, Pulli Kolam, Straight Line Kolam, or Curved Kolam based on dot/contour counts, similarity, and complexity.
- **Derived metrics** — complexity score, grid size estimate, design style, difficulty level, drawing method, and estimated drawing time.
- **AI recreation** — rebuilds the Kolam from detected dots using a k-nearest-neighbor connection graph, rendered as smooth Bezier-curved threads.
- **Original vs. recreated comparison** — generates a color difference heatmap and a structural match score between the uploaded photo and its recreation.
- **Analysis history** — every analysis is saved to a local SQLite database and shown in the sidebar, with an option to clear history.
- **PDF report export** — download a formatted report containing the original/recreated/heatmap images, full metrics table, and identified design principles.

### 🪔 Kolam Generation
- Procedurally generate a brand-new, original Kolam pattern from a symmetric dot grid (not derived from any uploaded image).
- Configurable grid size, symmetry mode (4-fold, 2-fold vertical, or 2-fold horizontal mirror), and pattern density.
- Download the generated design as a PNG.

### 📷 Flexible Input
- Upload an image file (JPG/JPEG/PNG) or capture one directly from your camera.

---

## 🗂 Project Structure

```
.
├── app.py                  # Streamlit UI and main analysis pipeline
├── image_processing.py     # Grayscale, noise removal, edges, contours, dot/region detection, diff heatmap
├── feature_extraction.py   # Symmetry, pattern classification, complexity, difficulty, etc.
├── kolam_recreation.py     # k-NN based Kolam recreation with curved thread rendering
├── kolam_generator.py      # Procedural generation of new symmetric Kolam designs
├── report_generator.py     # PDF report generation (ReportLab)
├── database.py             # SQLite storage for analysis history
└── kolam_history.db        # SQLite database (created automatically on first run)
```

---

## 🛠 Tech Stack

| Purpose            | Library                     |
|---------------------|------------------------------|
| Web UI              | [Streamlit](https://streamlit.io/) |
| Image processing    | [OpenCV](https://opencv.org/) (`opencv-python`) |
| Numerical operations | NumPy |
| Data handling        | pandas |
| Database            | SQLite (via Python's built-in `sqlite3`) |
| PDF reports         | [ReportLab](https://www.reportlab.com/) |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+

### Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install streamlit opencv-python numpy pandas reportlab
```

### Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---


## 🖼️ Screenshots

The screenshots below document the project/application workflow and supporting work.  
All images are stored in the [`screenshots/`](./screenshots/) folder so they render correctly when the repository is pushed to GitHub.

### 🎨 Kolam Design Principle Analyzer

#### 1. Application Home & Input
![Kolam Analyzer Home](./screenshots/11-kolam-analyzer-home.png)

The main dashboard provides options to upload a Kolam image, capture one from a camera, or generate a new original Kolam design.

#### 2. Image Processing Results
![Kolam Image Processing Results](./screenshots/12-kolam-image-processing-results.png)

The application displays the original image, grayscale conversion, and noise-removed output as part of the preprocessing pipeline.

#### 3. Edge, Contour & Dot Detection
![Kolam Edge Contours and Dots](./screenshots/13-kolam-edge-contours-dots.png)

The processed output visualizes edge detection, detected contours, and the detected structural dots used for further analysis.

### 📚 Supporting Course & Development Screenshots


---

## 📖 How It Works

1. **Upload or capture** a Kolam image.
2. The image is converted to **grayscale**, **denoised**, and passed through **Canny edge detection**.
3. **Contours** and **dots** are detected and counted; if no explicit dots are found, the app looks for the enclosed loop shapes typical of continuous-thread Kolams instead.
4. **Symmetry** is scored across horizontal, vertical, and rotational axes.
5. These measurements feed into **pattern classification** (Sikku / Pulli / Straight Line / Curved), plus complexity, difficulty, grid size, and confidence scores.
6. The detected dots are used to **recreate** the Kolam using a k-nearest-neighbor thread graph rendered with curved strokes.
7. The original and recreated versions are compared structurally to produce a **difference heatmap** and **match score**.
8. Results are displayed on an interactive dashboard, saved to the **analysis history**, and can be exported as a **PDF report**.

---

## 📊 Metrics Explained

| Metric | Description |
|---|---|
| **Pattern Type** | Classified Kolam style (Sikku / Pulli / Straight Line / Curved) |
| **Design Style** | Human-readable style label derived from pattern type |
| **Grid Size** | Estimated dot grid dimensions (`n x n`) |
| **Symmetry** | Horizontal-flip similarity rating |
| **Full Symmetry Type** | Combined horizontal/vertical/rotational symmetry classification |
| **Complexity** | Score (0–100) based on dot and contour density |
| **Difficulty** | Easy / Medium / Hard, derived from complexity |
| **Confidence** | Weighted score combining similarity and complexity |
| **Structural Match Score** | How closely the AI recreation matches the original's structure |

---

## 📝 Notes

- Analysis history is stored locally in `kolam_history.db` (SQLite) and persists between sessions.
- Generated PDF reports are created in a temporary file and streamed for download; they are not stored on disk afterward.
- This project is intended for the study and appreciation of traditional Tamil Kolam art through computer vision.

---

## 🙏 Acknowledgements

Inspired by the traditional Tamil art of **Kolam** — geometric line drawings created around a grid of dots, traditionally drawn by hand at the entrances of South Indian homes.
