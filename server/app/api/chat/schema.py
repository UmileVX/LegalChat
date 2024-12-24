from typing import List
from pydantic import BaseModel
from llama_index.core.llms import MessageRole


class _Message(BaseModel):
    role: MessageRole
    content: str


class ChatData(BaseModel):
    messages: List[_Message]
