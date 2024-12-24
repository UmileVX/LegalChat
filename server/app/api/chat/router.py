from fastapi import APIRouter, Depends, HTTPException, Request, status, BackgroundTasks
from llama_index.core.chat_engine.types import BaseChatEngine

# custom modules
from app.engine import get_chat_engine, get_custom_chat_engine
from app.engine.query_filter import generate_filters
from app.utils.logging import Logger
from app.utils.events import EventCallbackHandler

from .response import LegalChatStreamResponse
from .model import ChatData


chat_router = r = APIRouter()
logger = Logger()


@r.post("")
async def chat(
    request: Request,
    data: ChatData,
    background_tasks: BackgroundTasks,
    # chat_engine: BaseChatEngine = Depends(get_chat_engine),
):
    try:
        last_message_content = data.get_last_message_content()
        messages = data.get_history_messages()

        doc_ids = data.get_chat_document_ids()
        filters = generate_filters(doc_ids)
        params = data.data or {}
        logger.log_info(
            f"Creating chat engine with filters: {str(filters)}",
        )
        event_handler = EventCallbackHandler()
        chat_engine = get_custom_chat_engine(last_message_content, messages, verbose=False)
        response = chat_engine.astream_chat(last_message_content, messages)

        return LegalChatStreamResponse(
            request, event_handler, response, data, background_tasks
        )
    except Exception as e:
        logger.log_error("Error in chat engine", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in chat engine: {e}",
        ) from e
