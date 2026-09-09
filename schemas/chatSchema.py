import os
from pydantic import BaseModel, Field
from typing import Annotated

class ChatSchema(BaseModel):
    company: str
    year: str
    inquiry: str