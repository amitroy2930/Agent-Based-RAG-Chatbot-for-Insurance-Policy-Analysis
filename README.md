# Agent-Based RAG Chatbot for Insurance Policy Analysis

This project develops a scalable and accurate Retrieval-Augmented Generation (RAG) system designed to process and analyze extensive insurance policy documents. By integrating advanced techniques like Adaptive & Corrective RAG, a web search agent, and a hallucination check, the system provides reliable and factually grounded answers to complex user queries.

-----

### Features

  * **Scalable Document Processing:** Efficiently ingests and processes large volumes of insurance policy documents (over 500 pages) from various PDFs.
  * **Adaptive & Corrective RAG:** Enhances retrieval accuracy by dynamically restructuring user queries and filtering out irrelevant document chunks, ensuring that only the most pertinent information is used for generation.
  * **Real-time Information Retrieval:** Integrates a **Web Search Agent** using Tavily Search to fetch up-to-date information that may not be present in the policy documents.
  * **Hallucination Prevention:** Incorporates a **Hallucination Check** mechanism that validates the generated response against the retrieved documents, significantly reducing misinformation.
  * **Structured Output:** Provides well-structured, comprehensive, and coherent answers based on a multi-step refinement process.

-----

### Core Architecture

The system operates on a sophisticated, multi-step RAG pipeline to generate high-quality responses:

1.  **Query Restructuring:** The initial user query is refined and broken down into multiple structured sub-queries to improve document retrieval.
2.  **Document Retrieval:** The system retrieves relevant documents from a vector store based on the restructured queries.
3.  **Document Refinement:** An LLM refines the retrieved documents by assigning relevance scores and filtering out irrelevant information.
4.  **Final Response Generation:** A comprehensive answer is generated using the refined, relevant document chunks.
5.  **Hallucination Check:** The final response is validated to ensure it is factually grounded and free of fabricated information.

-----

### Tech Stack

  * **Languages:** Python
  * **Frameworks:** LangChain, LangGraph, AutoGen, LlamaIndex, FastAPI
  * **Large Language Models (LLMs):** OpenAI (GPT-4o, text-embedding-3-small), LLaMA, GPT, Ollama
  * **Vector Databases:** ChromaDB, FAISS
  * **Utilities:** LlamaParse (for PDF to Markdown conversion), TavilySearchResults (for web search), Prompt Engineering, LoRA, QLoRA
  * **Environment:** Azure (for OpenAI services)

-----

### Setup and Installation

1.  **Clone the repository:**
    ```
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  **Create a virtual environment:**
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```
4.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your API keys.
    ```
    LLAMA_CLOUD_API_KEY="..."
    AZURE_OPENAI_API_KEY="..."
    AZURE_OPENAI_ENDPOINT="..."
    TAVILY_API_KEY="..."
    openai_api_key="..."
    ```
5.  **Process Policy Documents:**
    Run the data ingestion notebook to convert PDFs into structured chunks and populate the ChromaDB vector store.
6.  **Run the Chatbot:**
    Execute the main notebook or a FastAPI application to interact with the RAG chatbot.

-----

### Usage

To get an answer for a query, simply call the `generate_response` function:

```python
query = "Does HDFC ERGO Optima Secure cover AYUSH treatment?"
generate_response(query)
```

The output will be saved to a Markdown file in the `./data/generated_output/` directory if the response is verified as factual.
