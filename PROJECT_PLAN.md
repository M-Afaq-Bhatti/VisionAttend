# VisionAttend - Project Plan

## 1. Overview & Objectives
Build a professional, MVP-grade AI-powered employee attendance and access monitoring system (VisionAttend). The system handles face detection, face recognition, anti-spoofing/liveness verification, and attendance event logging, wrapped in a polished Streamlit dashboard.

## 2. High-Level Architecture
The system is divided into modular components:
- **CV Pipeline**: Handles video frames, detects faces, aligns them, extracts embeddings, and runs anti-spoofing checks.
- **Attendance Engine**: Processes recognized faces and liveness scores, applies cooldown rules, and logs attendance.
- **Database Layer**: Manages employee profiles, face embeddings, attendance events, and security events.
- **Streamlit Dashboard**: A decoupled frontend that consumes the backend services for administration and live monitoring.

## 3. ML & CV Pipeline Plan
1. **Face Detection & Alignment**: Use `retinaface` or `mtcnn` for robust face detection and facial landmark extraction.
2. **Quality Gate**: Check face size, pose, and confidence. Reject invalid frames.
3. **Face Embedding**: Use pretrained ArcFace/InsightFace (`insightface` Python package) to generate robust 512-D embeddings.
4. **Identity Matching**: Compute cosine similarity between the current frame's embedding and registered embeddings. Use configurable thresholds.
5. **Anti-Spoofing**: Implement an abstract `BaseAntiSpoofModel` interface. For MVP, use a lightweight pre-trained liveness detection model (e.g., Silent-Face-Anti-Spoofing or similar open-source model) to detect 2D spoof attacks.

## 4. Database Schema Plan (SQLAlchemy)
- **Employee**: id, employee_code, full_name, department, position, email, status, created_at, updated_at
- **FaceProfile**: id, employee_id, embedding, model_name, embedding_dimension, created_at
- **AttendanceEvent**: id, employee_id, event_type (check-in), timestamp, similarity_score, liveness_score, source, status
- **RecognitionEvent**: id, timestamp, employee_id (nullable), identity_status, similarity_score, quality_status, liveness_status, source
- **SecurityEvent**: id, timestamp, event_type, employee_id (nullable), confidence, source, metadata

## 5. Streamlit UI Architecture
- Sidebar Navigation: Overview, Live Monitoring, Employees, Attendance, Security Events, Analytics, Model Evaluation, Settings.
- Styling: Custom CSS for a professional, enterprise-grade dark/navy theme.
- Caching: Use `@st.cache_resource` for CV models and DB engines to prevent reloading.

## 6. Testing Strategy
- **Unit Tests**: `pytest` for similarity calculation, attendance rules, database CRUD operations.
- **Integration Tests**: End-to-end processing of a mock frame through the CV pipeline and Attendance Engine.
- **Evaluation**: Utility scripts to evaluate the recognition model on a controlled dataset, producing metrics like False Acceptance Rate (FAR) and False Rejection Rate (FRR).

## 7. Dependency Plan
- **CV**: `opencv-python`, `insightface`, `onnxruntime` (for inference), `albumentations` or `numpy` for augmentations/processing.
- **DB**: `sqlalchemy`, `psycopg2-binary` (or `asyncpg`), `alembic` for migrations.
- **UI**: `streamlit`, `pandas`, `plotly` for analytics charts.
- **Core**: `pydantic` for data validation, `python-dotenv` for config.
- **Testing**: `pytest`, `pytest-cov`.

## 8. Implementation Roadmap
- [x] **PHASE 1**: Research & Planning
- [ ] **PHASE 2**: Environment & Skeleton (Virtual env, config, DB layer, basic Streamlit app)
- [ ] **PHASE 3**: Face Recognition (Detection, ArcFace embeddings, enrollment)
- [ ] **PHASE 4**: Anti-Spoofing (Liveness detection integration)
- [ ] **PHASE 5**: Attendance Engine (Rules, persistence, logging)
- [ ] **PHASE 6**: Streamlit Dashboard (UI implementation, pages)
- [ ] **PHASE 7**: Evaluation (Evaluation scripts, metric reports)
- [ ] **PHASE 8**: Polish (Refactoring, error handling, visual polish)
- [ ] **PHASE 9**: Final Verification (End-to-end testing)
