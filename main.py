"""
Main entry point for the ai_call_agent FastAPI application.
Configures routes, WebSocket connections for Twilio, and lifecycle events.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings, configure_langsmith
configure_langsmith()

from telephony.twilio_handler import twilio_handler
from mock_api.orders import router as orders_router
from mock_api.appointments import router as appointments_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that handles startup and shutdown logic.
    Initializes/verifies SQLite and ChromaDB directories.
    """
    logger.info("Starting ai_call_agent server...")
    
    # Ensure database parent directories exist
    db_dir = os.path.dirname(settings.appointments_db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    
    # Initialize the database engine & tables
    from mock_api.appointments import init_db
    init_db()
    
    # Initialize the RAG FAQ retriever connection
    from rag.retriever import faq_retriever
    faq_retriever.initialize()
    
    logger.info(f"SQLite path verified at: {settings.appointments_db_path}")
    logger.info(f"ChromaDB persistence directory verified at: {settings.chroma_persist_dir}")
    
    # Run startup verification checks
    import aiohttp
    
    # 1. Verify Ollama is reachable
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{settings.ollama_base_url}/api/tags", timeout=3.0) as resp:
                if resp.status == 200:
                    logger.info("Startup check: Ollama is reachable and active.")
                else:
                    logger.warning(f"Startup check: Ollama returned status {resp.status} at {settings.ollama_base_url}")
    except Exception as e:
        logger.warning(f"Startup check: Ollama is unreachable at {settings.ollama_base_url}: {e}")
        
    # 2. Verify Deepgram API key is valid
    try:
        headers = {"Authorization": f"Token {settings.deepgram_api_key}"}
        async with aiohttp.ClientSession() as session:
            # We use HEAD request to avoid downloading excess data
            async with session.head("https://api.deepgram.com/v1/projects", headers=headers, timeout=3.0) as resp:
                if resp.status == 200:
                    logger.info("Startup check: Deepgram API key is valid.")
                else:
                    logger.warning(f"Startup check: Deepgram API verification returned status {resp.status} (invalid API key or subscription issues?)")
    except Exception as e:
        logger.warning(f"Startup check: Deepgram key verification failed: {e}")
        
    # 3. Verify ChromaDB FAQ collection exists with records
    try:
        if faq_retriever.is_initialized() and faq_retriever._collection is not None:
            count = faq_retriever._collection.count()
            if count > 0:
                logger.info(f"Startup check: ChromaDB FAQ collection contains {count} records.")
            else:
                logger.warning("Startup check: ChromaDB FAQ collection is empty. Did you run the FAQ ingestor?")
        else:
            logger.warning("Startup check: ChromaDB FAQ collection is not initialized.")
    except Exception as e:
        logger.warning(f"Startup check: ChromaDB collection verification failed: {e}")
        
    yield
    logger.info("Shutting down ai_call_agent server...")

app = FastAPI(
    title="Hospital Voice AI Agent",
    description="Automated customer service voice agent for hospital clients",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include mock APIs
app.include_router(orders_router, prefix="/orders", tags=["Orders"])
app.include_router(appointments_router, prefix="/appointments", tags=["Appointments"])

@app.get("/health")
async def health_check():
    """Health check endpoint to verify server is running."""
    return {"status": "ok", "port": 5000}

@app.post("/webhook")
async def twilio_webhook(request: Request):
    """
    Webhook endpoint for Twilio initial call setup.
    Dynamically constructs the WebSocket URL using the incoming Host header
    and passes the caller's phone number as a query parameter.
    """
    form_data = await request.form()
    caller_number = form_data.get("From", "Unknown")
    host = request.headers.get("host", f"localhost:{settings.app_port}")
    
    # TwiML Response to connect media stream
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{host}{settings.websocket_path}?caller_number={caller_number}" />
  </Connect>
</Response>
"""
    return Response(content=twiml, media_type="application/xml")

@app.websocket("/twilio")
async def twilio_websocket(websocket: WebSocket):
    """WebSocket endpoint for handling Twilio Media Streams."""
    await twilio_handler(websocket)

@app.get("/metrics")
async def get_metrics():
    """Endpoint to return the last 10 turn latencies as JSON."""
    from telephony.twilio_handler import RECENT_LATENCIES
    return list(RECENT_LATENCIES)

@app.get("/debug/last-call")
async def get_last_call_debug():
    """Returns the diagnostics dictionary from the most recent completed call."""
    import telephony.twilio_handler as handler
    return handler.LAST_CALL_DIAGNOSTICS
