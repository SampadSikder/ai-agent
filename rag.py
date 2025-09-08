import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# LlamaIndex
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from models.chat_model import Classification

# 1) LLM (LangChain) — requires langchain-google-genai + google-genai
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
structured_llm = llm.with_structured_output(Classification)

# 2) Embeddings for LlamaIndex (set BEFORE loading the index to avoid OpenAI fallback)
Settings.embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004")

# 3) Load your persisted LlamaIndex
PERSIST_DIR = "storage/imdb_llamaindex"
storage_ctx = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_ctx)

# 4) Build a retriever
retriever = index.as_retriever(similarity_top_k=5)

# 5) Retrieval adapters
def retrieve_examples(query: str):
    return retriever.retrieve(query)

def format_examples(nodes) -> str:
    parts = []
    for i, n in enumerate(nodes, 1):
        m = n.metadata or {}
        parts.append(
            f"""Example {i}
Text: {n.text}
Sentiment: {m.get('sentiment')}
Aggressiveness: {m.get('aggressiveness')}
Language: {m.get('language')}
---"""
        )
    return "\n".join(parts)

prompt = ChatPromptTemplate.from_template(
    """
You are an information extractor that must output ONLY the fields required by the `Classification` schema.

First, study these similar labeled examples (they are authoritative hints on how to classify):
{few_shot_examples}

Now classify the new passage. Be consistent with examples when possible.

Passage:
{input}

Instructions:
- Use the `Classification` schema with exact fields.
- sentiment should be either 'positive' or 'negative' for IMDB-like text (or 'neutral'/'mixed' if truly ambiguous).
- aggressiveness is an integer in [1..10].
- language is a short code like "en".
- Do not add fields.
"""
)
print(prompt.input_variables)

chain = (
    {
        "few_shot_examples": RunnableLambda(lambda x: format_examples(retrieve_examples(x))),
        "input": RunnablePassthrough()
    }
    | prompt
    | structured_llm
)

if __name__ == "__main__":
    tests = [
        "An absolute masterpiece with stellar performances and a heartfelt story!",
        "This was painfully bad. The script is garbage and the acting is worse!",
        "Not perfect, but I enjoyed most of it.",
        "Worst fucking shit I've ever seen!!! What a waste of time and money!!!",
        "Not bad, not good"
    ]
    for t in tests:
        print("\nINPUT:", t)
        print("OUTPUT:", chain.invoke(t))
