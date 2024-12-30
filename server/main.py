from dotenv import load_dotenv

load_dotenv()

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import llama_index.core

# custom modules
from app.api.chat.router import chat_router
from app.settings import init_settings
from app.utils.logging import Logger


app = FastAPI()
logger = Logger()

# Default to 'dev' (development) if not set
environment = os.getenv("ENVIRONMENT", "dev")
if environment not in {"dev", "prod"}:
    environment = "dev"

init_settings()

if environment == "dev":
    llama_index.core.set_global_handler("simple")

    logger.log_warning("Running in development mode - allowing CORS for all origins")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(chat_router, prefix="/api/chat")


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=8000, reload=True)
