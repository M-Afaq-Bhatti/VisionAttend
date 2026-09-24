# Engineering Decisions

## 1. Computer Vision Models
- **Face Detection and Recognition**: `insightface` (ArcFace). Pretrained, highly accurate, industry-standard embeddings.
- **Anti-Spoofing**: Lightweight open-source liveness models via ONNX (e.g., Silent-Face-Anti-Spoofing). Keeps dependencies minimal and inference fast on CPU/GPU.
- **Model Format**: ONNX. Ensures cross-platform compatibility and allows easy fallback from GPU (CUDA) to CPU.

## 2. Database (PostgreSQL via SQLAlchemy)
- **Choice**: PostgreSQL is the standard for production systems. SQLAlchemy ORM is used for database interactions.
- **Development Profile**: For ease of setup (if PostgreSQL is not available locally), the system will support SQLite as a local fallback through a simple configuration switch (e.g., `DATABASE_URL=sqlite:///./local.db`).

## 3. UI Framework
- **Choice**: Streamlit. Allows rapid development of a functional dashboard with Python.
- **Architecture**: The UI code will be strictly separated from the business logic. UI pages will call functions from the `src.services` and `src.cv` modules, rather than embedding complex CV code inside Streamlit rendering loops.
- **Performance**: Will use `@st.cache_resource` for loading models to avoid reloading heavy CV models on every Streamlit interaction.

## 4. Quality Gates and Unknown Handling
- **Quality Gate**: Before attempting recognition, frames will be analyzed for face size, pose, and clarity.
- **Unknown Handling**: Instead of forcing a match, similarity scores below the configurable threshold will be logged as 'UNKNOWN' security events.

## 5. Environment and Deployment
- **Dependency Management**: Standard `requirements.txt` with specific versions.
- **Dotenv**: Using `.env` for secrets/configurations.
- **Separation of Concerns**: A clear `src/` layout separating `cv`, `database`, `attendance`, `antispoof`, and `pipelines`.
