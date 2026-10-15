from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel
import os
import tempfile
import base64
from typing import Optional

import config
from services.pipeline import AssistantPipeline
from services.gemini import GeminiError

import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Student AI Assistant", version="1.0.0")

# Mount static files from config-resolved absolute paths (works in dev and
# in the packaged EXE, regardless of the current working directory).
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

# Global pipeline instance - offline by default
pipeline = AssistantPipeline(force_offline=True)

# Security
security = HTTPBearer(auto_error=False)

# Request models
class ChatRequest(BaseModel):
    message: str

class PDFOpenRequest(BaseModel):
    path: str

class PDFCloseRequest(BaseModel):

    pass

# PDF state per session (simple in-memory, isolated per process)
_active_pdf: Optional[dict] = None

def get_active_pdf():
    return _active_pdf

def set_active_pdf(state: dict):
    global _active_pdf
    _active_pdf = state

# Health endpoints
@app.get("/api/health")
async def health():
    return {"status": "ok", "mode": pipeline.mode}

@app.get("/api/status")
async def status():
    return {"mode": pipeline.mode}

# Chat endpoint
@app.post("/api/chat")
async def chat(request: ChatRequest):
    message = (request.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message must not be empty")
    try:
        response = pipeline.reply(message)
        return {"response": response.text, "mode": pipeline.mode}
    except GeminiError as exc:
        # Safe, sanitised category only - never the raw error or API key.
        logger.warning("Chat online failure: %s", exc)
        return {"response": "Online AI is unavailable right now. Please try again later.", "mode": "offline"}
    except Exception:
        logger.exception("Chat endpoint failed")
        raise HTTPException(status_code=500, detail="Internal error")

# PDF endpoints
@app.post("/pdf/open")
async def pdf_open(req: PDFOpenRequest):
    global _active_pdf
    path = req.path
    
    # Validate path - prevent path traversal
    if ".." in path or path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid path")
    
    # Check file exists and is .pdf
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    
    if not path.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Not a PDF file")
    
    # Check file size (limit to 50MB)
    file_size = os.path.getsize(path)
    if file_size > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")
    
    try:
        result = pipeline.load_pdf(path)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "PDF load failed"))
        
        set_active_pdf({
            "path": path,
            "pages": result.get("pages", 0),
            "text_length": result.get("text_length", 0),
            "subject_hints": result.get("subject_hints", []),
        })
        return {
            "success": True,
            "pages": result.get("pages", 0),
            "characters": result.get("text_length", 0),
            "subject_hints": result.get("subject_hints", []),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF error: {str(e)}")

@app.post("/pdf/close")
async def pdf_close():
    global _active_pdf
    _active_pdf = None
    return {"success": True, "message": "PDF session cleared"}

@app.post("/pdf/info")
async def pdf_info():
    active = get_active_pdf()
    if not active:
        raise HTTPException(status_code=400, detail="No active PDF session")
    return {
        "path": active.get("path"),
        "pages": active.get("pages"),
        "characters": active.get("text_length"),
        "subject_hints": active.get("subject_hints", []),
    }

@app.post("/pdf/ask")
async def pdf_ask(req: dict):
    question = req.get("question", "")
    active = get_active_pdf()
    if not active:
        return {"response": "No active PDF session. Use /pdf open first."}
    
    try:
        response = pipeline.query_pdf(question)
        return {"response": response}
    except Exception as e:
        return {"response": f"Error querying PDF: {str(e)}"}

@app.post("/pdf/extract")
async def pdf_extract():
    active = get_active_pdf()
    if not active:
        return {"response": "No active PDF session", "summary": ""}
    
    # Return summary of extracted text
    text_len = active.get("text_length", 0)
    pages = active.get("pages", 0)
    subject_hints = active.get("subject_hints", [])
    
    # Read first part of the text for summary
    summary = ""
    try:
        with open(active["path"], "rb") as f:
            # Use pypdf to extract first 300 chars
            from pypdf import PdfReader
            reader = PdfReader(active["path"])
            text_parts = []
            for page in reader.pages[:3]:  # First 3 pages
                if page.extract_text():
                    text_parts.append(page.extract_text())
            if text_parts:
                summary = " ".join(text_parts)[:300] + ("..." if len(" ".join(text_parts)) > 300 else "")
    except Exception:
        summary = "Could not extract text summary"
    
    return {
        "summary": summary,
        "total_characters": text_len,
        "pages": pages,
        "subject_hints": subject_hints,
    }

# Gemini mode endpoint
@app.post("/mode/switch")
async def mode_switch(enable_online: bool = False):
    global pipeline
    try:
        if enable_online:
            # Initialize with Gemini online mode
            pipeline = AssistantPipeline(force_offline=False)
        else:
            pipeline = AssistantPipeline(force_offline=True)
        return {"mode": pipeline.mode, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mode switch error: {str(e)}")

# Serve the web interface
@app.get("/", include_in_schema=False)
async def get():
    return FileResponse(str(config.INDEX_TEMPLATE))

# Serve favicon if needed
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = config.STATIC_DIR / "img" / "favicon.ico"
    return FileResponse(str(favicon_path)) if favicon_path.exists() else None