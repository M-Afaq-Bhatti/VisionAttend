# VisionAttend

**Smart Workforce Attendance & Access Monitoring System**

VisionAttend is a professional MVP-grade AI-powered employee attendance and access monitoring system. 
It performs real-time face detection, recognition, and anti-spoofing to securely log attendance events.

## Features
- Real-time face detection & alignment.
- High-accuracy face recognition using ArcFace embeddings.
- Anti-spoofing/liveness verification to prevent presentation attacks.
- Configurable attendance engine (cooldowns, duplicate suppression).
- Professional Streamlit dashboard for monitoring and management.

## Project Structure
```text
visionattend/
├── app/               # Streamlit application (frontend)
├── src/               # Core business logic (CV, Attendance, DB)
├── tests/             # Pytest unit and integration tests
├── data/              # Storage for local data, enrollment, evaluation
├── scripts/           # Utility scripts (init DB, evaluation)
├── configs/           # Configuration files
├── reports/           # Evaluation metric outputs
└── docs/              # Additional documentation
```

## Setup Instructions

### 1. Prerequisites
- Python 3.10+
- (Optional) NVIDIA GPU with CUDA for faster inference.

### 2. Environment Setup
```bash
python -m venv .venv

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configuration
Copy `.env.example` to `.env` and fill in the required values.

### 4. Running the Application
```bash
streamlit run app/streamlit_app.py
```
