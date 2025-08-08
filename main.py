from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
import google.generativeai as genai
# ---------- Pinecone imports changed ----------
from pinecone import Pinecone, ServerlessSpec
# ----------------------------------------------
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import uvicorn
import re
import logging
import time
import spacy
from concurrent.futures import ThreadPoolExecutor
import asyncio
import requests

# ---------------- Logging & Threadpool ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure parallel processing
PROCESSING_THREADS = 4
executor = ThreadPoolExecutor(max_workers=PROCESSING_THREADS)

# ---------------- Load spaCy ----------------
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded spaCy model for text processing")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {str(e)}")
    raise SystemExit(1)

# ---------------- Load environment variables ----------------
load_dotenv()

def get_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        logger.error(f"Missing required environment variable: {name}")
        raise SystemExit(1)
    return value

GEMINI_API_KEY = get_env_var("GEMINI_API_KEY")
PINECONE_API_KEY = get_env_var("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_env_var("PINECONE_INDEX_NAME")
HACKRX_TOKEN = get_env_var("HACKRX_TOKEN")
EMBEDDING_DIMENSION = 768  # Gemini embedding dimension

# ---------------- Initialize Gemini ----------------
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    logger.info("Gemini initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {str(e)}")
    raise SystemExit(1)

# ---------------- Initialize Pinecone (NEW STYLE) ----------------
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # list existing indexes (the Pinecone object returns a structure with .names())
    existing_names = []
    try:
        # pc.list_indexes() in some SDKs returns object with .names() as in docs; handle robustly
        li = pc.list_indexes()
        if hasattr(li, "names"):
            existing_names = li.names()
        elif isinstance(li, (list, tuple)):
            existing_names = list(li)
        else:
            # try to coerce
            existing_names = list(li)
    except Exception:
        # fallback to empty list on unexpected shape
        existing_names = []

    recreate_index = False
    if PINECONE_INDEX_NAME in existing_names:
        try:
            index_info = pc.describe_index(PINECONE_INDEX_NAME)
            # index_info may be dict-like with 'dimension'
            idx_dim = None
            if isinstance(index_info, dict):
                idx_dim = index_info.get("dimension")
            elif hasattr(index_info, "dimension"):
                idx_dim = getattr(index_info, "dimension")
            if idx_dim and int(idx_dim) != EMBEDDING_DIMENSION:
                logger.warning(f"Existing index has wrong dimension ({idx_dim} vs {EMBEDDING_DIMENSION}). Deleting...")
                pc.delete_index(PINECONE_INDEX_NAME)
                recreate_index = True
                time.sleep(5)
        except Exception:
            # If describe fails, attempt recreate
            recreate_index = True

    if PINECONE_INDEX_NAME not in existing_names or recreate_index:
        logger.info(f"Creating index {PINECONE_INDEX_NAME} with dimension {EMBEDDING_DIMENSION}...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        logger.info("Waiting briefly for index initialization...")
        time.sleep(5)

    # Create an Index client from the Pinecone client
    index = pc.Index(PINECONE_INDEX_NAME)

    try:
        stats = index.describe_index_stats()
        logger.info(f"Using Pinecone index: {PINECONE_INDEX_NAME}")
        logger.info(f"Index stats - vector count: {stats.get('total_vector_count', 0)}")
    except Exception:
        logger.info("Index ready (stats unreadable).")

except Exception as e:
    logger.error(f"Pinecone initialization failed: {str(e)}")
    raise SystemExit(1)

# ---------------- FastAPI setup ----------------
app = FastAPI(title="PDF Question Answering Platform")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------- Utility functions (original logic preserved) ----------------
def truncate_by_bytes(text: str, max_bytes: int = 40000) -> str:
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode('utf-8', errors='ignore').rstrip()

async def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        logger.info(f"Fast text extraction from: {pdf_path}")
        text = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: "\n".join(
                page.extract_text() for page in PdfReader(pdf_path).pages if page.extract_text()
            )
        )

        if not text.strip():
            raise ValueError("PDF contains no extractable text")

        logger.info(f"Extracted {len(text.split())} words")
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise HTTPException(400, f"PDF processing error: {str(e)}")

async def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    def process_chunk(sentences):
        chunks = []
        current_chunk = []
        current_size = 0

        for sent in sentences:
            word_count = len(sent.split())
            if current_size + word_count > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(sent)
            current_size += word_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    try:
        doc = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: nlp(text[:1000000])
        )

        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        if not sentences:
            return []

        batch_size = max(1, len(sentences) // PROCESSING_THREADS + 1)
        batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]

        tasks = [
            asyncio.get_event_loop().run_in_executor(executor, process_chunk, batch)
            for batch in batches
        ]

        results = await asyncio.gather(*tasks)
        chunks = [chunk for sublist in results for chunk in sublist]

        logger.info(f"Created {len(chunks)} chunks using {PROCESSING_THREADS} threads")
        return chunks

    except Exception as e:
        logger.error(f"Chunking error: {str(e)}")
        raise HTTPException(500, f"Text processing error: {str(e)}")

def get_embedding(text: str) -> List[float]:
    try:
        if not text.strip():
            raise ValueError("Empty text provided")

        res = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document",
            title="PDF Content"
        )

        vector = res.get("embedding") or res.get("data")
        if isinstance(vector, dict) and "embedding" in vector:
            vector = vector["embedding"]
        if not isinstance(vector, list):
            raise ValueError("Invalid vector format")
        # optionally warn if len != EMBEDDING_DIMENSION
        if len(vector) != EMBEDDING_DIMENSION:
            logger.warning(f"Embedding dimension mismatch: got {len(vector)} (expected {EMBEDDING_DIMENSION})")
        return vector
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(500, f"Embedding generation failed: {str(e)}")

async def index_chunks(chunks: List[str]):
    try:
        logger.info(f"Background indexing started for {len(chunks)} chunks")

        def process_batch(batch):
            vectors = []
            for i, chunk in enumerate(batch):
                try:
                    vector = get_embedding(chunk)
                    vectors.append({
                        "id": f"chunk-{time.time_ns()}-{i}",
                        "values": vector,
                        "metadata": {"text": truncate_by_bytes(chunk)}
                    })
                except Exception as e:
                    logger.warning(f"Skipping chunk {i}: {str(e)}")
            return vectors

        batch_size = 50
        all_vectors = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectors = await asyncio.get_event_loop().run_in_executor(
                executor,
                process_batch,
                batch
            )
            all_vectors.extend(vectors)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

        if all_vectors:
            index.upsert(vectors=all_vectors)
            logger.info(f"Indexed {len(all_vectors)} vectors")

    except Exception as e:
        logger.error(f"Background indexing failed: {str(e)}")

def is_complex_question(question: str) -> bool:
    complexity_triggers = [
        "calculate", "how much", "total", "combined",
        "over time", "after", "before", "difference",
        "compared to", "step by step", "process"
    ]
    return any(trigger in question.lower() for trigger in complexity_triggers)

def search_similar_chunks(query: str, top_k: int = 7) -> List[str]:
    try:
        query_vector = get_embedding(query)
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )

        matches = results.get("matches", []) or results.get("matches", [])
        chunks_with_scores = [
            (match["metadata"]["text"], match.get("score") or match.get("similarity", None))
            for match in matches
        ]

        logger.info(f"Top chunk scores: {[score for _, score in chunks_with_scores]}")
        return [chunk for chunk, _ in chunks_with_scores]

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(500, f"Search failed: {str(e)}")

def build_prompt(question: str, contexts: List[str]) -> str:
    context_text = "\n\n".join(f"### Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts))

    prompt = f"""
You are an expert document analyst. Answer the question using ONLY the provided context.
When information needs to be combined from multiple contexts:
1. Identify relevant elements from each context
2. Explain how they connect logically
3. Provide the synthesized answer

If the answer requires calculation:
- Show your step-by-step reasoning
- State any assumptions
- Provide the final numeric answer

Structure your response:
1. Direct answer to the question
2. Reasoning path (with context references)
3. Confidence score 1-10 (based on context support)

Contexts:
{context_text}

Question: {question}

Response:
"""
    logger.debug(f"Generated reasoning prompt: {prompt[:500]}...")
    return prompt

def ask_gemini(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", None) or (response.get("text") if isinstance(response, dict) else None)
        if not text:
            raise ValueError("Empty response from Gemini")
        return text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {str(e)}")
        return f"Error generating answer: {str(e)}"

# ---------------- API Endpoints (original ones preserved) ----------------

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing: {file.filename} (size: {getattr(file, 'size', 0)/1024:.1f}KB)")

        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(400, "Only PDF files are allowed")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            text = await extract_text_from_pdf(tmp_path)
            chunks = await chunk_text(text)

            if not chunks:
                raise HTTPException(400, "No valid text chunks created")

            chunks = chunks[:1000] if len(chunks) > 1000 else chunks
            # Background indexing like original
            asyncio.create_task(index_chunks(chunks))

            return {"success": True, "message": "PDF is being processed in background"}

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(500, f"Processing error: {str(e)}")

@app.post("/api/ask")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"Received question: {question}")

        if not question.strip():
            raise HTTPException(400, "Question cannot be empty")

        top_k = 10 if is_complex_question(question) else 5
        logger.info(f"Using top_k={top_k} for {'complex' if top_k == 10 else 'simple'} question")

        relevant_chunks = search_similar_chunks(question, top_k=top_k)

        if not relevant_chunks:
            return {"answer": "No relevant information found in the document"}

        prompt = build_prompt(question, relevant_chunks)
        answer = ask_gemini(prompt)

        return {"answer": answer}

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Question error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error answering question: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------------- HackRX combined endpoint (uses original logic) ----------------
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

def check_auth(authorization: Optional[str]):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ", 1)[1].strip()
    if token != HACKRX_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

@app.post("/api/v1/hackrx/run", response_model=RunResponse)
async def run_submission(req: RunRequest, authorization: Optional[str] = Header(None)):
    check_auth(authorization)

    try:
        logger.info(f"Downloading document: {req.documents}")
        r = requests.get(req.documents, timeout=30)
        if r.status_code != 200:
            raise HTTPException(400, f"Failed to download PDF (status {r.status_code})")
    except requests.RequestException as e:
        logger.error(f"Download failed: {str(e)}")
        raise HTTPException(400, f"Failed to download PDF: {str(e)}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(r.content)
        pdf_path = tmp.name

    try:
        text = await extract_text_from_pdf(pdf_path)
        chunks = await chunk_text(text)

        if not chunks:
            raise HTTPException(400, "No valid text chunks created from the provided PDF")

        chunks = chunks[:1000] if len(chunks) > 1000 else chunks

        # Index (await so search will find these chunks immediately)
        await index_chunks(chunks)

        # Answer questions using original ask logic
        answers = []
        for q in req.questions:
            top_k = 10 if is_complex_question(q) else 5
            logger.info(f"Question: {q} -> using top_k={top_k}")
            contexts = search_similar_chunks(q, top_k=top_k)
            if not contexts:
                answers.append("No relevant information found in the document")
                continue
            prompt = build_prompt(q, contexts)
            ans = ask_gemini(prompt)
            answers.append(ans)

        return RunResponse(answers=answers)

    finally:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)

# ---------------- Run ----------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
