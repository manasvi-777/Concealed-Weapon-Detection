# Concealed Weapon Detection (CWD) - YOLOv8

## Project Overview

This repository contains the complete pipeline for a **Concealed Weapon Detection** solution using the YOLOv8 object detection framework. The pipeline covers dataset preparation, training, evaluation, and deployment with a Streamlit web application.

## Project Structure & Files

```bash
CWD/
├── CWD_1epoch.ipynb        # Jupyter Notebook - Full training and evaluation log
├── data.yaml               # YOLOv8 Dataset Configuration
├── best.pt                 # Final trained model weights
├── yolov8n.pt              # Pre-trained Nano model used for transfer learning
├── convert-labeltoYolo.py  # Python script for converting annotations
├── extract_frames.py       # Python script to extract images from video data
├── splitintoTRAINandVAL.py # Python script for creating 80/20 train/val splits
├── streamlit-app/
│   ├── app7.py             # Streamlit Web Application for Detection
│   └── requirements.txt    # Python dependencies for Streamlit app
├── data/
│   └── UCLA_Thermal_Imaging_Dataset/   # Raw Dataset Folder – **NOT UPLOADED TO GIT**
└── runs/                   # YOLOv8 Training Logs – **IGNORED**
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/concealed-weapon-detection.git
cd concealed-weapon-detection/CWD
```

### 2. Install Dependencies

You can install the required dependencies for training and the Streamlit app using:

```bash
pip install -r streamlit-app/requirements.txt
```

For YOLOv8 installation:

```bash
pip install ultralytics
```

### 3. Dataset Setup

* Place the dataset inside the `data/` directory.
* Update the dataset configuration in `data.yaml` if needed.

⚠️ Note: The **UCLA Thermal Imaging Dataset** is not uploaded to GitHub due to size restrictions.

### 4. Training the Model

Run the training Jupyter notebook:

```bash
jupyter notebook CWD_1epoch.ipynb
```

Or train directly using YOLOv8 CLI:

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

### 5. Running the Streamlit Web App

To start the detection web app:

```bash
cd streamlit-app
streamlit run app7.py
```

## 📈 Model Performance Metrics

The model was fine-tuned for a limited number of epochs on the custom thermal dataset. The final validation run showed strong performance metrics on the validation set, especially concerning the **Mean Average Precision (mAP)** and high recall for the critical **Handgun class**.

| Metric           | Overall | Handgun Class | Person Class |
| ---------------- | ------- | ------------- | ------------ |
| **mAP@0.5**      | 0.970   | 0.946         | 0.994        |
| **mAP@0.5:0.95** | 0.677   | 0.701         | 0.653        |
| **Precision**    | 0.956   | 0.939         | 0.973        |
| **Recall**       | 0.948   | 0.905         | 0.990        |
| **F1 Score**     | 0.952   | N/A           | N/A          |

### Interpretation

* **High mAP@0.5 (0.970):** The model is highly effective at correctly drawing bounding boxes around objects with at least 50% overlap (IoU) between the prediction and ground truth.
* **Strong Recall (0.948):** The model successfully detected 94.8% of actual objects (Handguns and Persons), crucial for security applications.
* **Handgun mAP@0.5:0.95 (0.701):** The model maintained strong detection accuracy for Handguns even under stricter IoU thresholds.


