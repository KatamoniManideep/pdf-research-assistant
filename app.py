import os
import asyncio
from typing import AsyncGenerator, Any, cast
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# LangChain Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

from config import Config
from pydantic import SecretStr

# instantiate config to access instance attributes (fixes static analysis warnings)
config = Config()

app = FastAPI(title="PDF Research Assistant API")


hf_model = getattr(config, "HF_EMBEDDING_MODEL", None) or os.getenv("HF_EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=hf_model)
persist_directory = os.path.join(config.DATA_DIR, "chroma_db")

if os.path.exists(persist_directory):
    vector_store= Chroma(persist_directory=persist_directory,embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k":4})
else:
    vector_store=None
    retriever=None

google_key = getattr(config, "GOOGLE_API_KEY", None) or os.getenv("GOOGLE_API_KEY") or None
google_api_key: SecretStr | None = SecretStr(google_key) if google_key is not None else None

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=google_api_key,
)

session_store={}

def get_session_history(session_id:str)-> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id]= ChatMessageHistory()
    return session_store[session_id]


contextual_prompt= ChatPromptTemplate.from_messages([
    ("system","You are an advanced research assistant. Answer the user's questions using only the provided context below. If you do not know the answer, say that you do not know.\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human","{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs if retriever else lambda x: "",
        "question": lambda x: x["question"],
        "history": lambda x: x.get("history", [])
    }
    | contextual_prompt
    | llm
)

managed_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)


class QueryRequest(BaseModel):
    question: str
    session_id: str

async def generate_llm_stream(question: str,session_id: str)-> AsyncGenerator[str,None]:

    if not retriever:
        yield "Error: Vector store not initialized. Please ingest documents first."
        return
    try:
        docs = await retriever.ainvoke(question)
        context_text ="\n\n".join([doc.page_content for doc in docs])

        async for chunk in managed_rag_chain.astream(
            {"context": context_text, "question": question},
            config={"configurable": {"session_id": session_id}}
        ):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        yield f"\n[Internal Error: {str(e)}]"

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    
    return StreamingResponse(
        generate_llm_stream(request.question, request.session_id),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.py:app", host="127.0.0.1", port=8000, reload=True)