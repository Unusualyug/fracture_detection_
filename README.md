# 🩻 AI-Powered Fracture Detection

<p align="center">
  <img src="https://img.shields.io/badge/AI-Medical%20Imaging-0ea5e9?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Deep%20Learning-ResNet50-8b5cf6?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Frontend-React-61DAFB?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge" />
</p>

<p align="center">
  <strong>Deep Learning based X-Ray Fracture Detection System</strong>
</p>

<p align="center">
  Upload an X-Ray → AI analyzes the image → Receive a fracture prediction
</p>

<p align="center">
  <a href="https://github.com/Unusualyug/fracture_detection_">Repository</a>
</p>

---

## ✨ Overview

**Fracture Detection** is an end-to-end medical imaging application that uses **Deep Learning and Computer Vision** to classify X-Ray images for possible bone fractures.

The project combines a **ResNet50-based image classification model** with a **FastAPI inference backend** and a **React web interface**, creating a complete AI application rather than just a standalone ML model.

### 🔄 How It Works

```text
                    ┌──────────────────┐
                    │    X-Ray Image   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  React Frontend  │
                    │   Image Upload   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   FastAPI API    │
                    │ Image Validation │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    ResNet50      │
                    │  PyTorch Model   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Prediction     │
                    │ Fracture/Normal  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ React Result UI  │
                    └──────────────────┘
```

---

## 🚀 Key Features

* 🩻 **X-Ray Image Analysis**
* 🧠 **ResNet50 Deep Learning Model**
* ⚡ **FastAPI Inference API**
* ⚛️ **React Frontend**
* 🔬 **Binary Fracture Classification**
* 📊 **Confidence-based Prediction**
* 🐍 **PyTorch Model Inference**
* 🌐 **Full-stack AI Architecture**
* ☁️ **Deployment-ready architecture**
* 📦 **Separated frontend, backend and model components**

---

## 📊 Model Performance

The trained model was evaluated using multiple classification metrics.

| Metric                   |      Score |
| ------------------------ | ---------: |
| **Accuracy**             | **94.07%** |
| **ROC-AUC**              | **0.9813** |
| **Fracture Recall**      | **92.72%** |
| **Fracture Specificity** | **92.72%** |

### 🏆 Performance Summary

```text
Accuracy       ███████████████████░   94.07%
ROC-AUC        ████████████████████   98.13%
Recall         ███████████████████░   92.72%
Specificity    ███████████████████░   92.72%
```

> These results are based on the project's evaluation setup and should not be interpreted as clinical validation.

---

## 🧠 Deep Learning Model

The project uses **ResNet50**, a convolutional neural network architecture designed for image recognition.

### Why ResNet50?

ResNet50 uses **residual connections**, allowing deep neural networks to learn complex visual patterns while reducing optimization difficulties associated with very deep architectures.

For X-Ray analysis, the model learns image features that can help distinguish between:

```text
                 X-Ray Image
                      │
                      ▼
              ┌───────────────┐
              │   ResNet50     │
              │ Feature        │
              │ Extraction     │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Classification │
              │     Head       │
              └───────┬───────┘
                      │
               ┌──────┴──────┐
               ▼             ▼
          🟢 Normal      🔴 Fracture
```

---

## 🏗️ Project Architecture

```text
fracture_detection_
│
├── backend/
│   ├── app/
│   │   ├── best_model.pth
│   │   └── ...
│   │
│   ├── ...
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
│
├── ...
│
└── README.md
```

> Adjust the structure above if your repository contains additional or differently named directories.

---

## 🛠️ Technology Stack

### Artificial Intelligence

| Technology       | Purpose                  |
| ---------------- | ------------------------ |
| 🐍 Python        | ML & backend development |
| 🔥 PyTorch       | Deep Learning framework  |
| 🧠 ResNet50      | Image classification     |
| 🖼️ PIL / OpenCV | Image processing         |
| 📊 NumPy         | Numerical computation    |
| 📈 Scikit-learn  | Model evaluation         |

### Backend

| Technology | Purpose         |
| ---------- | --------------- |
| ⚡ FastAPI  | REST API        |
| 🚀 Uvicorn | ASGI server     |
| 📦 Python  | Backend runtime |

### Frontend

| Technology  | Purpose                          |
| ----------- | -------------------------------- |
| ⚛️ React    | User interface                   |
| 🎨 CSS      | Styling                          |
| 🌐 REST API | Frontend ↔ Backend communication |

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Unusualyug/fracture_detection_.git
cd fracture_detection_
```

---

## 2️⃣ Create a Python Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

---

## 3️⃣ Install Backend Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Backend

Start the FastAPI server using the project's backend entry point.

Example:

```bash
uvicorn app.main:app --reload
```

Once running, the API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI automatically provides interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# ⚛️ Running the Frontend

Move into the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The React application will then be available through the URL shown in your terminal.

---

# 🔌 API Workflow

The application follows a simple request-response pipeline:

```text
Frontend
   │
   │  Upload X-Ray
   ▼
FastAPI
   │
   │  Preprocess Image
   ▼
ResNet50
   │
   │  Model Inference
   ▼
Prediction
   │
   │  JSON Response
   ▼
Frontend
   │
   ▼
Display Result
```

### Example Response

```json
{
  "prediction": "Fracture",
  "confidence": 0.94
}
```

> The exact response fields depend on the current API implementation.

---

# 🧪 Model Evaluation

The model was evaluated using standard classification metrics.

### Accuracy

Measures the percentage of predictions classified correctly.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Recall

Measures how effectively the model identifies positive fracture cases.

```text
Recall = TP / (TP + FN)
```

### Specificity

Measures how effectively the model identifies non-fracture cases.

```text
Specificity = TN / (TN + FP)
```

### ROC-AUC

Measures the model's ability to distinguish between the two classes across classification thresholds.

---

# 📸 Application Workflow

```text
       ┌─────────────────┐
       │   Upload X-Ray  │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ Image Processing│
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │  ResNet50 Model │
       └────────┬────────┘
                │
          ┌─────┴─────┐
          ▼           ▼
      Fracture       Normal
          │           │
          └─────┬─────┘
                ▼
       ┌─────────────────┐
       │ Prediction UI   │
       └─────────────────┘
```

---

# 🌐 Deployment

The project can be deployed as separate services:

```text
                 INTERNET
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
     React Frontend        FastAPI Backend
          │                     │
          │                     ▼
          │                 ResNet50
          │                     │
          └──────────┬──────────┘
                     ▼
                 Prediction
```

This architecture makes it possible to independently deploy and scale the frontend and inference backend.

---

# 🔐 Medical Disclaimer

> ⚠️ **For Research and Educational Purposes Only**

This project is a **research/educational prototype** and is **not a medical device**.

Predictions generated by this system should **not** be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.

Any real-world clinical application would require appropriate medical validation, regulatory review, safety testing, and evaluation by qualified healthcare professionals.

---

# 🔮 Future Improvements

* [ ] Grad-CAM visual explainability
* [ ] Multi-class fracture classification
* [ ] Improved image preprocessing
* [ ] Model calibration
* [ ] Confidence threshold optimization
* [ ] Model monitoring
* [ ] Docker containerization
* [ ] Automated testing
* [ ] CI/CD pipeline
* [ ] Cloud-based inference
* [ ] Mobile-friendly interface
* [ ] Explainable AI dashboard

---

# 📚 Learning Outcomes

This project demonstrates practical experience with:

```text
Deep Learning
     ↓
Computer Vision
     ↓
Transfer Learning
     ↓
ResNet50
     ↓
Model Evaluation
     ↓
PyTorch Inference
     ↓
FastAPI
     ↓
REST APIs
     ↓
React
     ↓
Full-Stack AI Deployment
```

---

# 👨‍💻 Author

**Yug Lakhani**

Computer Science & Engineering

### GitHub

[![GitHub](https://img.shields.io/badge/GitHub-Unusualyug-181717?style=for-the-badge\&logo=github)](https://github.com/Unusualyug)

---

<p align="center">

### 🩻 Built with Deep Learning • PyTorch • FastAPI • React

**From X-Ray → AI Model → Prediction**

</p>

<p align="center">
  ⭐ If you find this project useful, consider giving it a star!
</p>
