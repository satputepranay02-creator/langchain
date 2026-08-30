import os

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
)
from langchain_postgres import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = PGVector(
    embeddings=embeddings_model,
    collection_name="langchain_text_file_loaded_chunks",
    connection=os.environ["DATABASE_URL"],
    use_jsonb=True,
)

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        max_new_tokens=512,
    )
)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_template(
    "Answer using only the context below.\n\n"
    "<context>\n{context}\n</context>\n\n"
    "Question: {question}"
)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


chain = (
    {"context": retriever | format_docs, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

app = FastAPI()


class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask(query: Query):
    return {"answer": chain.invoke(query.question)}
