# RLM Document Chatbot

An educational implementation of Recursive Language Models (RLM) for document analysis, based on the paper by Zhang, Kraska, and Khattab (2025).

**Paper**: [Recursive Language Models](https://arxiv.org/abs/2512.24601)
**GitHub**: [alexzhang13/rlm](https://github.com/alexzhang13/rlm)

## What is RLM?

RLM is an inference strategy where language models can decompose and recursively interact with input context of **unbounded length** through REPL environments.

### Key Insight

Instead of feeding all context directly to the LLM (which has limited context window), RLM:
1. Stores context as a `context` variable in a Python REPL
2. Lets the LLM write code to examine, search, and chunk the context
3. Enables recursive sub-LLM calls via `llm_query()`
4. Returns results via `FINAL(answer)`

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│                      User Query                          │
│            "What are the main themes?"                   │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    REPL Environment                      │
│  context = "[large document stored here]"                │
│  query = "What are the main themes?"                     │
│  llm_query(prompt) -> recursive LLM call                 │
│  FINAL(answer) -> return final answer                    │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    LLM writes code                       │
│                                                          │
│  ```python                                               │
│  # Examine structure                                     │
│  print(len(context))                                     │
│  print(context[:1000])                                   │
│  ```                                                     │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    Execute & Return                      │
│                                                          │
│  Output: 50000                                           │
│          [First 1000 chars...]                           │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    LLM continues...                      │
│                                                          │
│  ```python                                               │
│  # Search for themes                                     │
│  chunks = context.split('\n\n')                          │
│  for chunk in chunks[:5]:                                │
│      theme = llm_query(f"Theme of: {chunk}")             │
│      themes.append(theme)                                │
│  FINAL('\n'.join(themes))                                │
│  ```                                                     │
└─────────────────────────────────────────────────────────┘
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
│  - Document Upload (PDF, TXT, MD)                       │
│  - Chat Interface                                        │
│  - RLM Execution Trace Visualization                    │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                       │
│  - /upload - Document ingestion                         │
│  - /chat - Chat with RLM                                │
│  - /documents - List uploaded docs                      │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    RLM Engine                            │
│  - REPLEnvironment: Python execution sandbox            │
│  - RLMEngine: LLM ↔ REPL loop orchestration            │
│  - llm_query(): Recursive sub-LLM calls                 │
│  - FINAL(): Answer extraction mechanism                 │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    Document Store                        │
│  - ChromaDB for vector embeddings                       │
│  - Chunk-based retrieval                                │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    Ollama (llama3.1:8b)                 │
│  - Local LLM inference                                  │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
rlm-chatbot/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── run.sh                    # Start script
│
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── config.py            # Configuration
│   │
│   ├── rlm/                 # RLM Core Implementation
│   │   ├── __init__.py
│   │   ├── base.py          # RLMStep, RLMResult, REPLState
│   │   ├── environment.py   # REPL environment with llm_query/FINAL
│   │   ├── engine.py        # Main LLM ↔ REPL execution loop
│   │   └── summarizer.py    # Recursive document summarizer
│   │
│   ├── documents/
│   │   ├── __init__.py
│   │   ├── processor.py     # Document parsing & chunking
│   │   └── store.py         # ChromaDB vector store
│   │
│   └── llm/
│       ├── __init__.py
│       └── ollama_client.py # Ollama API wrapper
│
└── frontend/
    └── app.py               # Streamlit UI
```

## Prerequisites

- Python 3.10+
- Ollama installed with llama3.1:8b model

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Ollama is running**:
   ```bash
   ollama serve
   ```

3. **Start the application**:
   ```bash
   ./run.sh
   ```

   Or manually:
   ```bash
   # Terminal 1 - Backend
   uvicorn backend.main:app --reload --port 8000

   # Terminal 2 - Frontend
   streamlit run frontend/app.py
   ```

4. **Open your browser** to http://localhost:8501

## Usage

1. **Upload Documents**: Use the sidebar to upload PDF, TXT, or MD files
2. **Ask Questions**: Chat with your documents using natural language
3. **Watch RLM in Action**: See the code execution and REPL interaction

## RLM Execution Trace

The UI shows each step of the RLM process:
- **CODE**: Python code the LLM generated
- **OUTPUT**: Result from executing code in REPL
- **SUB_CALL**: Recursive `llm_query()` invocations
- **FINAL**: The returned answer via `FINAL()`

## Key RLM Concepts

### 1. Context as Variable
```python
# Context is NOT in the prompt - it's a variable
context = "[your large document here]"
print(len(context))  # 500000 characters
```

### 2. Strategic Access
```python
# LLM decides how to examine context
print(context[:1000])  # Peek at start
re.findall(r'Chapter \d+', context)  # Search structure
chunks = context.split('\n\n')  # Break into parts
```

### 3. Recursive Calls
```python
# Process chunks with sub-LLM calls
for chunk in chunks[:5]:
    summary = llm_query(f"Summarize: {chunk}")
    summaries.append(summary)
```

### 4. Final Answer
```python
# Return when confident
FINAL("The main themes are: ...")
# Or return a variable
FINAL_VAR("answer_buffer")
```

## Configuration

Edit `backend/config.py` to tune:
- `max_iterations`: Maximum code execution loops (default: 10)
- `max_depth`: Maximum recursion depth (default: 3)
- `chunk_size`: Document chunk size (default: 1000)

## References

- Zhang, A.L., Kraska, T., & Khattab, O. (2025). Recursive Language Models. arXiv:2512.24601
- [Blog Post by Alex Zhang](https://alexzhang13.github.io/blog/2025/rlm/)
- [Official RLM Implementation](https://github.com/alexzhang13/rlm)

## License

MIT License - Feel free to use for learning and experimentation.
