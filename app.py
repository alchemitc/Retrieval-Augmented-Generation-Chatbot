# app.py

# ─── Override sqlite3 with pysqlite3 ─────────────────────────────────────────────
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# ─── Standard imports ─────────────────────────────────────────────────────────────
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ─── Load environment variables ────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")

# ─── FastAPI setup ────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Build retriever ──────────────────────────────────────────────────────────────
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=api_key
)
vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings,
    collection_name="gigfinder"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ─── Define prompt template ───────────────────────────────────────────────────────
prompt = PromptTemplate(
    template="""
You are the Gig Marketplace chatbot designed exclusively to answer questions about the gig marketplace platform and its services.

User's Question: {input}

Context: {context}

IMPORTANT INSTRUCTIONS:
1. ONLY answer questions directly related to the gig marketplace, freelancing, jobs, or platform functionality.
2. If the user asks about ANY topic outside gig marketplace matters (like celebrities, general knowledge, programming languages, database queries, technology not specific to the platform, etc.), respond with: "I'm designed to assist only with questions about our gig marketplace. Please ask me about finding jobs, posting gigs, platform features, or other marketplace-related topics."
3. When answering valid gig-related questions, respond clearly, concisely, and conversationally based on the provided context.
4. If the context isn't sufficient for a valid gig-related question, politely say so and offer relevant help with marketplace features.
""",
    input_variables=["context", "input"] 

# ─── Initialize LLM ───────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.3,
    api_key=api_key
)

# ─── Build the RAG chain ──────────────────────────────────────────────────────────
combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_docs_chain)

# ─── WebSocket endpoint ───────────────────────────────────────────────────────────
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            user_input = data.get("question")  # keep key from frontend as "question"
            if not user_input:
                await websocket.send_json({"error": "Missing 'question'"})
                continue

            # Pass "input" to match what the prompt and chain expects
            result = rag_chain.invoke({"input": user_input})
            await websocket.send_json({"answer": result["answer"]})
    except WebSocketDisconnect:
        print("Client disconnected")
# app.py

# ─── Override sqlite3 with pysqlite3 ─────────────────────────────────────────────
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# ─── Standard imports ─────────────────────────────────────────────────────────────
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ─── Load environment variables ────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")

# ─── FastAPI setup ────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Build retriever ──────────────────────────────────────────────────────────────
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=api_key
)
vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings,
    collection_name="gigfinder"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ─── Define prompt template ───────────────────────────────────────────────────────
prompt = PromptTemplate(
    template="""
You are the Gig Marketplace chatbot. Respond clearly, concisely, and conversationally.

User's Question: {input}

Context: {context}

If context isn't sufficient, politely say so and offer help.
""",
    input_variables=["context", "input"]  # changed "question" -> "input"
)

# ─── Initialize LLM ───────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.3,
    api_key=api_key
)

# ─── Build the RAG chain ──────────────────────────────────────────────────────────
combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_docs_chain)

# ─── WebSocket endpoint ───────────────────────────────────────────────────────────
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            user_input = data.get("question")  # keep key from frontend as "question"
            if not user_input:
                await websocket.send_json({"error": "Missing 'question'"})
                continue

            # Pass "input" to match what the prompt and chain expects
            result = rag_chain.invoke({"input": user_input})
            await websocket.send_json({"answer": result["answer"]})
    except WebSocketDisconnect:
        print("Client disconnected")
