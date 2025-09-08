import os
import re
import math
import random
import pandas as pd
from dotenv import load_dotenv



load_dotenv()

import kagglehub


from llama_index.core import Document as LIDocument, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter



INDEX_DIR = "indexes/imdb_faiss"
DATASET_SLUG = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"

from models.empirical import estimate_aggressiveness
from tqdm import tqdm


def main():
    path = kagglehub.dataset_download(DATASET_SLUG)
    csv_path = os.path.join(path, "IMDB Dataset.csv")
    if not os.path.exists(csv_path):
        for f in os.listdir(path):
            if f.lower().endswith(".csv"):
                csv_path = os.path.join(path, f); break
    print("Using CSV:", csv_path)
    
    df = pd.read_csv(csv_path).dropna(subset=["review", "sentiment"]).copy()

    # Optional: use 10k rows (balanced). Comment out to use full 50k.
    SAMPLE_PER_CLASS = 5000
    df_pos = df[df["sentiment"] == "positive"].sample(SAMPLE_PER_CLASS, random_state=42)
    df_neg = df[df["sentiment"] == "negative"].sample(SAMPLE_PER_CLASS, random_state=42)
    df = pd.concat([df_pos, df_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 3) Prepare LlamaIndex docs (with metadata)
    docs = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        text = str(row["review"])
        sentiment = str(row["sentiment"])
        docs.append(
            LIDocument(
                text=text,
                metadata={
                    "sentiment": sentiment,
                    "aggressiveness": estimate_aggressiveness(text),
                    "language": "en",
                },
            )
        )
    
    print(f"Loaded {len(docs)} documents.")

    # 4) Set embedding model and build index
    Settings.embed_model = GoogleGenAIEmbedding(model="text-embedding-004")  # uses GOOGLE_API_KEY This uses default embedding setting of Google
    Settings.node_parser = SentenceSplitter(
    chunk_size=1000,      # like RecursiveCharacterTextSplitter(chunk_size=1000)
    chunk_overlap=200,    # like overlap=200
    )
    index = VectorStoreIndex.from_documents(docs)
    print(f"Built LlamaIndex with {len(docs)} docs.")

    # 5) Persist to disk (simple vector store + docstore)
    PERSIST_DIR = "storage/imdb_llamaindex"
    os.makedirs(PERSIST_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"Saved LlamaIndex to {PERSIST_DIR} with {len(docs)} docs.")


if __name__ == "__main__":
    main()