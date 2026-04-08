import os


class Config:
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    TESTING = os.getenv("FLASK_TESTING", "false").lower() == "true"
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
    JSON_SORT_KEYS = False
