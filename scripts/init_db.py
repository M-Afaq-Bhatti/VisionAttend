import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.session import init_db

if __name__ == "__main__":
    print("Initializing Database...")
    init_db()
    print("Database Initialized Successfully.")
