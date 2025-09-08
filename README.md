
# ai-agent

## Setup Instructions

### 1. Environment Setup
- Clone the repository.
- Install dependencies:
	```bash
	pip install -r requirements.txt
	```
- Create a `.env` file in the project root and add your API keys (e.g., `GOOGLE_API_KEY`).

### 2. Setting Up the Database
- Ensure you have your data source (e.g., IMDB dataset) available.
- If using a local database, configure the connection string in your `.env` or config file as needed.

### 3. Creating the Vector Database
- Run the script to build the vector store (example for IMDB):
	```bash
	python build_vectorstore_imdb.py
	```
- This will process your data and store embeddings in the `storage/` directory.

### 4. Running Retrieval-Augmented Generation (RAG)
- To run the RAG pipeline:
	```bash
	python rag.py
	```
- The script will use the vector database to retrieve relevant documents and generate answers using your LLM.

---

**Note:**
- Make sure your `.env` file is not tracked by git (already in `.gitignore`).
- For custom datasets or models, adjust the scripts as needed.
