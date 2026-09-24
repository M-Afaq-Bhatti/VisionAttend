from pydantic_settings import BaseSettings
from typing import Literal
import os

class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite:///./visionattend.db"
    
    # App config
    log_level: str = "INFO"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    
    # CV config
    recognition_threshold: float = 0.45
    liveness_threshold: float = 0.80
    
    # Attendance rules
    attendance_cooldown_minutes: int = 5
    snapshot_retention_days: int = 7
    
    # Modes
    demo_mode: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
