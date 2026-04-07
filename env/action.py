from pydantic import BaseModel
from typing import Optional

class Action(BaseModel):
    action_type: str
    email_id: int
    classification: Optional[str] = None
    reply_text: Optional[str] = None
