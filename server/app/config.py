import os

DATA_DIR = "data"
STATIC_DIR = os.getenv("STATIC_DIR", "static")

# check if for debug
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
