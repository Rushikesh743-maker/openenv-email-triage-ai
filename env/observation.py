from pydantic import BaseModel
from typing import List

class Email(BaseModel):
    id: int
    sender: str
    subject: str
    body: str

class Observation(BaseModel):
    inbox: List[Email]
    current_email_index: int
