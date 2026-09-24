from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)

engine = create_engine(
    settings.database_url, 
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    from src.database.models import Base
    logger.info(f"Initializing database at {settings.database_url}")
    Base.metadata.create_all(bind=engine)
