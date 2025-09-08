from pydantic import BaseModel, Field
from typing import Literal


class Classification(BaseModel):
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="The sentiment of the text"
    )
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(description="The language the text is written in")

