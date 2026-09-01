# Architecture — LangChain RAG Pipeline

End-to-end Retrieval-Augmented Generation pipeline over a PDF. Ingest a PDF via
LLM-vision parsing, chunk + embed locally, store vectors in Postgres/PGVector
with incremental indexing, retrieve, generate answers with Groq, and evaluate
with a keyword sanity check plus RAGAS metrics.

## Flow

```mermaid
flowchart TD
    subgraph INGEST["Ingestion"]
        PDF["PDF file<br/>langchain-raw-data.pdf"]
        MUPDF["PyMuPDF<br/>render each page -> PNG @150dpi"]
        VISION["ChatGoogleGenerativeAI<br/>gemini-3.6-flash (vision)<br/>page PNG -> markdown + [IMAGE] notes"]
        DOCS["Documents<br/>1 per page (5 pages)"]
        PDF --> MUPDF --> VISION --> DOCS
    end

    subgraph PREP["Chunk + Embed"]
        SPLIT["RecursiveCharacterTextSplitter<br/>chunk_size=1000, overlap=200"]
        EMB["HuggingFaceEmbeddings<br/>all-MiniLM-L6-v2 (local)"]
        DOCS --> SPLIT --> EMB
    end

    subgraph STORE["Vector Store"]
        PG["PGVector<br/>postgres:5432<br/>collection: langchain_text_file_loaded_chunks"]
        RM["SQLRecordManager<br/>incremental index, cleanup=full<br/>source_id_key=source"]
        EMB --> RM --> PG
    end

    subgraph QUERY["Retrieval + Generation"]
        RET["Retriever<br/>similarity, k=4"]
        PROMPT["ChatPromptTemplate<br/>answer from context only"]
        LLM["ChatGroq<br/>gpt-oss-20b (answers)"]
        OUT["Answer<br/>StrOutputParser"]
        PG --> RET --> PROMPT --> LLM --> OUT
    end

    subgraph EVAL["Evaluation"]
        KW["Keyword hit-rate check<br/>5 Q/keyword pairs"]
        RAGAS["RAGAS<br/>gpt-oss-120b judge<br/>faithfulness, answer_relevancy,<br/>context_precision, context_recall"]
        RET --> KW
        RET --> RAGAS
        LLM --> RAGAS
    end

    Q["User question"] --> RET
```

## Components

| Stage | Tool | Notes |
|-------|------|-------|
| PDF render | PyMuPDF | pure pip, no poppler/tesseract |
| Page parse | Gemini 3.6 Flash vision | text + chart/image description |
| Split | RecursiveCharacterTextSplitter | 1000 / 200 |
| Embed | all-MiniLM-L6-v2 | local, 384-dim |
| Store | PGVector on Postgres | jsonb metadata |
| Index | SQLRecordManager | dedup + full cleanup |
| Retrieve | PGVector retriever | k=4 |
| Generate | Groq gpt-oss-20b | temperature=0 |
| Eval | keyword check + RAGAS | judge = gpt-oss-120b |

## Config / secrets
`.env`: `GOOGLE_API_KEY` (vision), `GROQ_API_KEY` (generation + judge). Postgres
connection hardcoded in notebook: `postgresql+psycopg://postgres:0000@localhost:5432/postgres`.
