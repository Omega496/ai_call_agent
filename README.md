# Hospital Voice AI Receptionist Agent

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.110.0+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestrator-orange?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Deepgram](https://img.shields.io/badge/Deepgram-STT%20%26%20TTS-black?logo=deepgram&logoColor=white)](https://deepgram.com)
[![Twilio](https://img.shields.io/badge/Twilio-Telephony-F22F46?logo=twilio&logoColor=white)](https://twilio.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-blueviolet?logo=database&logoColor=white)](https://trychroma.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-white?logo=ollama&logoColor=black)](https://ollama.com)

An automated customer-service voice agent designed for hospital clients to handle inbound phone calls end-to-end without human intervention. The system converts raw caller speech to text (STT), routes context through a LangGraph state machine, retrieves medical context using a RAG pipeline, generates reception responses with a local LLM, and streams text-to-speech (TTS) back to the caller in real-time.

Designed for low-latency voice interactions:
- **FAQ Turns**: $\le$ 1.5s latency (Speech-to-first-audio-chunk)
- **RAG & Tool Calling Turns**: $\le$ 3.0s latency

---

## 🛠️ Core Features

* **Real-time Bidirectional Audio Streaming**: Direct integration with Twilio Media Streams using asynchronous WebSockets (`twilio_receiver` and `twilio_sender` tasks).
* **Interruption Handling (Barge-in)**: Instantly halts active TTS streams and clears Twilio's audio buffer when the caller interrupts mid-response.
* **Intelligent Intent Classification**: Dynamically routes calls to appropriate logic nodes (FAQ, Scheduling, Orders, Escalation) using a local LLM.
* **Semantic Search (RAG)**: Embeds and stores hospital FAQ information in ChromaDB using Ollama embeddings to answer receptionist questions.
* **Multi-Turn Slot Filling (Appointment Booking)**: Collects patient name, target medical department, and preferred dates before calling the appointment booking tool.
* **Live Order Tracking**: Automatically extracts order IDs from conversations and interacts with a mock API to fetch order statuses.
* **Silence Watchdog Watcher**: Monitors the audio stream for extended caller silence, prompting twice with warnings before gracefully hanging up the call.
* **Call Escalation**: Automatically redirects active calls to a human agent number using the Twilio REST API when required.

---

## 🏗️ Architecture

```
Caller ──> Twilio (Virtual Number) ──> TwiML Bin / Connect Webhook
                                               │
                                               ▼
                                   FastAPI WebSocket (/twilio)
                                               │
                                 ┌─────────────┴─────────────┐
                                 ▼                           ▼
                        [twilio_receiver]             [twilio_sender]
                          Inbound Audio                Outbound Audio
                     (base64 mulaw 8kHz Mono)     (base64 mulaw 8kHz Mono)
                                 │                           ▲
                                 ▼                           │
                       Convert to Linear16 PCM               │
                                 │                           │
                                 ▼                           │
                        Deepgram STT (VAD)                   │
                       (nova-3-medical API)                  │
                                 │                           │
                            speech_final                     │
                                 ▼                           │
                      LangGraph State Machine                │
                    ┌─────────────────────────┐              │
                    │   classify_intent node  │              │
                    └────────────┬────────────┘              │
                                 │ (conditional routing)     │
            ┌────────────────────┼────────────────────┐      │
            ▼                    ▼                    ▼      │
       [faq_node]        [order_status_node]   [scheduler_node]   │
      RAG & ChromaDB       Mock GET /orders     Slot Filling │
            └────────────────────┬────────────────────┘      │
                                 ▼                           │
                           Ollama Model                      │
                      (nemotron-3-super:120b)                │
                                 │                           │
                            stream=True                      │
                                 ▼                           │
                            TTS Pipeline                     │
                       Deepgram TTS (Speak) ─────────────────┘
                      (aura-2-andromeda-en)
```

---

## 📂 Project Structure

```
hospital-voice-agent/
├── main.py                       # FastAPI application & entry point
├── core/
│   ├── config.py                 # Pydantic environment configurations
│   ├── latency.py                # Latency profile trackers & handlers
│   └── state.py                  # TypedDict definition for AgentState
├── telephony/
│   ├── twilio_handler.py         # WebSocket router, watchdog, & sender/receiver tasks
│   └── audio_utils.py            # mulaw <-> linear16 PCM conversion helpers
├── agent/
│   ├── graph.py                  # LangGraph StateGraph pipeline compilation
│   ├── nodes/
│   │   ├── classifier.py         # Intent classification logic node
│   │   ├── faq.py                # FAQ retrieval & RAG response node
│   │   ├── order_status.py       # Order query and API integration node
│   │   ├── scheduler.py          # Appointment reservation slot filling node
│   │   └── escalation.py         # Human agent handoff and TwiML dial redirection
│   └── tools/
│       ├── order_tool.py         # GET /orders/{order_id} langchain tool
│       └── appointment_tool.py   # POST /appointments langchain tool
├── rag/
│   ├── ingestor.py               # Raw document chunking & ChromaDB indexing
│   ├── retriever.py              # ChromaDB vector query interface
│   └── data/
│       └── faqs.md               # Raw hospital knowledge database document
├── mock_api/
│   ├── orders.py                 # Mock router endpoints for patient orders
│   └── appointments.py           # SQLite db configuration & appointment endpoints
├── tests/
│   ├── test_api.py               # REST API and endpoint testing
│   ├── test_graph.py             # LangGraph node routing tests
│   ├── test_rag.py               # RAG document parsing & retrieval tests
│   └── test_tools.py             # Tool invocation mock testing
├── pyproject.toml                # Project configurations & dependencies
├── requirements.txt              # Auto-generated dependency mappings
└── .env.example                  # Environment configuration template
```

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` and configure your API credentials and endpoints:

```bash
# Twilio Configuration
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_PHONE_NUMBER=+1xxxxxxxxxx
HUMAN_AGENT_NUMBER=+1xxxxxxxxxx

# Deepgram AI API
DEEPGRAM_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Ollama Local Models
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=nemotron-3-super:120b

# Embeddings & ChromaDB Configurations
CHROMA_PERSIST_DIR=./rag/chroma_db
CHROMA_COLLECTION_NAME=hospital_faqs
EMBEDDING_MODEL=nomic-embed-text:v1.5

# Mock Server Endpoint Configuration
MOCK_API_BASE_URL=http://localhost:5000
APPOINTMENTS_DB_PATH=./mock_api/appointments.db

# LangSmith Tracing
LANGCHAIN_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=hospital-voice-agent
LANGCHAIN_TRACING_V2=true

# Application Properties
APP_HOST=0.0.0.0
APP_PORT=5000
WEBSOCKET_PATH=/twilio
LOG_LEVEL=INFO
```

---

## 🚀 Getting Started

### 1. Prerequisite Installations
Ensure you have **Python 3.11+** installed. Download and start [Ollama](https://ollama.com) on your local machine, then pull the required models:
```bash
ollama pull nemotron-3-super:120b
ollama pull nomic-embed-text:v1.5
```

### 2. Set Up Virtual Environment & Dependencies
Initialize and activate your Python environment:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```
Install the package along with its execution dependencies:
```bash
pip install -r requirements.txt
# Alternatively, install using setuptools:
pip install .
```

### 3. Ingest FAQ Documentation
Populate the local vector store with the synthetic hospital FAQ database:
```bash
python -m rag.ingestor
```

### 4. Run the FastAPI Application
Start the uvicorn development server:
```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

### 5. Open Tunnel & Wire Telephony Webhook
In a new terminal window, initialize an ngrok tunnel to expose your local FastAPI port:
```bash
ngrok http 5000
```
Copy the generated secure `https` URL and create a **TwiML Bin** inside your Twilio Console pointing to the WebSocket stream path:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://YOUR-NGROK-URL/twilio" />
  </Connect>
</Response>
```
Route inbound calls for your active Twilio phone number to this TwiML Bin.

---

## 🧪 Testing

The test suite contains comprehensive unit and integration testing across mock API endpoints, RAG queries, tool routing, and state machine transitions.

To run tests in isolation:
```bash
pytest tests/ -v
```
