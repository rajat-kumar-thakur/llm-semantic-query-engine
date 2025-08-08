from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
import google.generativeai as genai
import pinecone
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import uvicorn
import re
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate environment variables
def get_env_var(name):
    value = os.getenv(name)
    if not value:
        logger.error(f"Missing required environment variable: {name}")
        exit(1)
    return value

GEMINI_API_KEY = get_env_var("GEMINI_API_KEY")
PINECONE_API_KEY = get_env_var("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_env_var("PINECONE_INDEX_NAME")
EMBEDDING_DIMENSION = 768  # Gemini embedding dimension 

# Initialize Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-pro-latest")  # Updated model name
    logger.info("Gemini initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {str(e)}")
    exit(1)

# Initialize Pinecone
try:
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists and has correct dimension
    recreate_index = False
    if PINECONE_INDEX_NAME in pc.list_indexes().names():
        index_info = pc.describe_index(PINECONE_INDEX_NAME)
        if index_info.dimension != EMBEDDING_DIMENSION:
            logger.warning(f"Existing index has wrong dimension ({index_info.dimension} vs {EMBEDDING_DIMENSION}). Deleting...")
            pc.delete_index(PINECONE_INDEX_NAME)
            recreate_index = True
            time.sleep(10)  # Wait for deletion
    
    # Create index if needed
    if PINECONE_INDEX_NAME not in pc.list_indexes().names() or recreate_index:
        logger.info(f"Creating index {PINECONE_INDEX_NAME} with dimension {EMBEDDING_DIMENSION}...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=pinecone.ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        logger.info("Waiting 1 minute for index initialization...")
        time.sleep(60)

    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    logger.info(f"Using Pinecone index: {PINECONE_INDEX_NAME}")
    logger.info(f"Index stats - dimension: {stats['dimension']}, vector count: {stats['total_vector_count']}")
except Exception as e:
    logger.error(f"Pinecone initialization failed: {str(e)}")
    exit(1)

# FastAPI setup
app = FastAPI(title="PDF Question Answering Platform")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Utility functions
def is_valid_url(url: str) -> bool:
    regex = re.compile(
        r'^(?:http|ftp)s?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        logger.info(f"Extracting text from: {pdf_path}")
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        
        if not text.strip():
            raise ValueError("PDF contains no extractable text")
            
        logger.info(f"Extracted {len(text.split())} words from document")
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise HTTPException(400, f"PDF processing error: {str(e)}")

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_len = len(word) + 1  # +1 for space
        if current_size + word_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
            
        current_chunk.append(word)
        current_size += word_len
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    logger.info(f"Created {len(chunks)} text chunks")
    return chunks

def get_embedding(text: str) -> List[float]:
    try:
        if not text.strip():
            raise ValueError("Empty text provided")
            
        res = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document",
            title="PDF Content"  # Optional but recommended
        )
        
        vector = res["embedding"]
        
        # Validate the embedding
        if not isinstance(vector, list):
            raise ValueError("Embedding is not a list")
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(f"Invalid vector dimension: {len(vector)} (expected {EMBEDDING_DIMENSION})")
        if not all(isinstance(x, (int, float)) for x in vector):
            raise ValueError("Vector contains non-numeric values")
            
        return vector
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(500, f"Embedding generation failed: {str(e)}")

def index_chunks(chunks: List[str]):
    try:
        vectors = []
        for i, chunk in enumerate(chunks):
            try:
                vector = get_embedding(chunk)
                
                vectors.append({
                    "id": f"chunk-{i}",
                    "values": vector,
                    "metadata": {"text": chunk[:1000]}  # Limit metadata size
                })
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                continue
        
        if not vectors:
            raise ValueError("No valid vectors to index")
        
        # Batch upsert with smaller batches and delays
        batch_size = 50
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            try:
                index.upsert(vectors=batch)
                logger.info(f"Indexed batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
                time.sleep(0.2)  # Small delay between batches
            except Exception as e:
                logger.error(f"Failed to index batch {i//batch_size}: {str(e)}")
                raise
        
        return True
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        raise HTTPException(500, f"Indexing failed: {str(e)}")

def search_similar_chunks(query: str, top_k: int = 5) -> List[str]:
    try:
        query_vector = get_embedding(query)
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        chunks = [match["metadata"]["text"] for match in results["matches"]]
        logger.info(f"Found {len(chunks)} relevant chunks for query: {query}")
        return chunks
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(500, f"Search failed: {str(e)}")

def build_prompt(question: str, contexts: List[str]) -> str:
    context_text = "\n\n".join(f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts))
    prompt = f"""
You are an expert document analyst. Use ONLY the provided context to answer the question.
If the context doesn't contain the answer, say "I couldn't find a clear answer in the document."

Contexts:
{context_text}

Question: {question}
Answer in 2-3 sentences:
"""
    logger.debug(f"Generated prompt: {prompt[:200]}...")  # Log first 200 chars
    return prompt

def ask_gemini(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        if not response.text:
            raise ValueError("Empty response from Gemini")
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {str(e)}")
        return f"Error generating answer: {str(e)}"

# API Endpoints
@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file upload: {file.filename}")
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(400, "Only PDF files are allowed")
            
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
            logger.info(f"Saved temporary file at: {tmp_path}")

        # Process PDF
        try:
            text = extract_text_from_pdf(tmp_path)
            chunks = chunk_text(text)
            
            if not chunks:
                raise HTTPException(400, "No valid text chunks created")
            
            # Limit number of chunks to prevent overload
            if len(chunks) > 1000:
                chunks = chunks[:1000]
                logger.warning("Truncated to first 1000 chunks")
                
            success = index_chunks(chunks)
            if not success:
                raise HTTPException(500, "Partial indexing failure")
                
            return {"success": True, "message": "PDF processed successfully!"}
            
        finally:
            # Cleanup temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                logger.info(f"Removed temporary file: {tmp_path}")
                
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error processing PDF: {str(e)}")

@app.post("/api/ask")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"Received question: {question}")
        
        if not question.strip():
            raise HTTPException(400, "Question cannot be empty")
            
        relevant_chunks = search_similar_chunks(question)
        
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

# Frontend
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)