
import os
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate


llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

from models.chat_model import Classification

tagging_prompt = ChatPromptTemplate.from_template(
  """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

structured_llm = llm.with_structured_output(Classification)

inp = "How are you doing today? I hope you have a really realllyyyyy a badd DAY!!"
prompt = tagging_prompt.invoke({"input": inp})
response = structured_llm.invoke(prompt)

print(response)