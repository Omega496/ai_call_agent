"""
Twilio WebSocket handler.
Manages bidirectional audio streams, receiver/sender tasks, and call state.
"""

import asyncio
import json
import logging
import re
import aiohttp
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage, AIMessage
from agent.nodes.escalation import transfer_call_to_human

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v1.types.listen_v1results import ListenV1Results

from agent.graph import run_agent_turn as run_agent_turn_graph
from core.config import settings
from core.state import AgentState, initial_state
from telephony.audio_utils import (
    decode_twilio_audio,
    encode_audio_for_twilio,
    build_clear_message,
    chunk_audio
)

logger = logging.getLogger(__name__)

import time
from collections import deque
from core.latency import LatencyTracker

RECENT_LATENCIES = deque(maxlen=10)

LAST_CALL_DIAGNOSTICS = None

class WebSocketDiagnostics:
    """Tracks WebSocket connection health metrics per call."""
    
    def __init__(self, call_sid: str):
        self.call_sid = call_sid
        self.connected_at = time.time()
        self.start_event_received = False
        self.audio_chunks_received = 0
        self.speech_final_count = 0
        self.tts_chunks_sent = 0
        self.errors = []
    
    def log_start_event(self):
        self.start_event_received = True
        logger.debug(f"[{self.call_sid}] Diagnostics: Twilio start event received.")
        
    def log_audio_chunk(self):
        self.audio_chunks_received += 1
        
    def log_speech_final(self, transcript: str):
        self.speech_final_count += 1
        logger.debug(f"[{self.call_sid}] Diagnostics: Speech final count is now {self.speech_final_count}.")
        
    def log_tts_chunk_sent(self):
        self.tts_chunks_sent += 1
        
    def log_error(self, error: str):
        self.errors.append(error)
        logger.error(f"[{self.call_sid}] Diagnostics Error: {error}")
    
    def diagnose(self) -> dict:
        """
        Return diagnostic dict with detected issues.
        
        Checks:
        - start_event_received is False after 5s → "TwiML misconfiguration suspected"
        - audio_chunks_received > 100 and speech_final_count == 0 → "VAD not triggering"
        - tts_chunks_sent == 0 and speech_final_count > 0 → "TTS pipeline not running"
        """
        issues = []
        elapsed = time.time() - self.connected_at
        
        if not self.start_event_received and elapsed > 5.0:
            issues.append("TwiML misconfiguration suspected")
            
        if self.audio_chunks_received > 100 and self.speech_final_count == 0:
            issues.append("VAD not triggering")
            
        if self.speech_final_count > 0 and self.tts_chunks_sent == 0:
            issues.append("TTS pipeline not running")
            
        return {
            "call_sid": self.call_sid,
            "connected_at": self.connected_at,
            "duration": elapsed,
            "start_event_received": self.start_event_received,
            "audio_chunks_received": self.audio_chunks_received,
            "speech_final_count": self.speech_final_count,
            "tts_chunks_sent": self.tts_chunks_sent,
            "errors": self.errors,
            "issues": issues,
            "status": "UNHEALTHY" if issues or self.errors else "HEALTHY"
        }
    
    def summary(self) -> str:
        """Human-readable summary for logging on call end."""
        d = self.diagnose()
        issues_str = "\n  - ".join(d["issues"]) if d["issues"] else "None"
        errors_str = "\n  - ".join(d["errors"]) if d["errors"] else "None"
        return (
            f"WebSocket Diagnostics Summary for Call {self.call_sid}:\n"
            f"  Status: {d['status']}\n"
            f"  Duration: {d['duration']:.1f}s\n"
            f"  Start Event Received: {d['start_event_received']}\n"
            f"  Audio Chunks Received: {d['audio_chunks_received']}\n"
            f"  Speech Final Events: {d['speech_final_count']}\n"
            f"  TTS Chunks Sent: {d['tts_chunks_sent']}\n"
            f"  Detected Issues: {issues_str}\n"
            f"  Logged Errors: {errors_str}"
        )

async def twilio_sender(websocket: WebSocket, audio_queue: asyncio.Queue, state: AgentState, diagnostics: WebSocketDiagnostics = None):
    """
    Outbound TTS task. Dequeues raw mulaw audio from the queue,
    encodes it to base64, and sends it to the Twilio WebSocket.
    """
    logger.info("Twilio sender task started.")
    stream_sid = state["stream_sid"]
    try:
        while True:
            chunk = await audio_queue.get()
            try:
                msg = encode_audio_for_twilio(stream_sid, chunk)
                await websocket.send_text(msg)
                if diagnostics:
                    diagnostics.log_tts_chunk_sent()
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected during sending.")
                if diagnostics:
                    diagnostics.log_error("Twilio sender WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error in twilio_sender websocket write: {e}")
                if diagnostics:
                    diagnostics.log_error(f"Twilio sender WebSocket write error: {e}")
                break
            finally:
                audio_queue.task_done()
    except asyncio.CancelledError:
        logger.info("Twilio sender task cancelled.")
    except Exception as e:
        logger.error(f"Unexpected error in twilio_sender: {e}")
        if diagnostics:
            diagnostics.log_error(f"Unexpected error in twilio_sender: {e}")

async def hangup_call(call_sid: str) -> None:
    """Hang up a Twilio call via REST API."""
    try:
        from twilio.rest import Client
        client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
        client.calls(call_sid).update(status="completed")
        logger.info(f"Successfully hung up call via Twilio REST API: {call_sid}")
    except Exception as e:
        logger.error(f"Error hanging up call via Twilio REST API: {e}", exc_info=True)

async def stream_response_to_caller(
    response_text: str,
    stream_sid: str,
    websocket: WebSocket,
    audio_queue: asyncio.Queue,
    state_ref: dict,
    t_speech_final: Optional[float] = None,
    speech_final_to_graph_start_ms: Optional[float] = None,
    graph_invocation_ms: Optional[float] = None,
    turn_num: Optional[int] = None,
    intent: Optional[str] = None,
    diagnostics: Optional[WebSocketDiagnostics] = None
) -> None:
    """
    Convert text response to audio and stream to caller via Twilio WebSocket.
    
    Uses Deepgram TTS REST API with streaming response.
    Audio format: mulaw 8kHz, container=none (required by Twilio).
    """
    state_ref["state"]["is_speaking"] = True
    t_text_ready = time.perf_counter()
    
    try:
        # Deepgram TTS endpoint
        url = (
            f"https://api.deepgram.com/v1/speak"
            f"?model=aura-2-andromeda-en"
            f"&encoding=mulaw"
            f"&sample_rate=8000"
            f"&container=none"
        )
        headers = {
            "Authorization": f"Token {settings.deepgram_api_key}",
            "Content-Type": "application/json"
        }
        payload = {"text": response_text}
        
        first_chunk = True
        # Stream audio response
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                async for chunk in resp.content.iter_chunked(8000):
                    # Check for barge-in before sending each chunk
                    if not state_ref["state"]["is_speaking"]:
                        break  # Barge-in detected — stop sending
                    
                    if first_chunk:
                        first_chunk = False
                        t_first_chunk = time.perf_counter()
                        if t_speech_final is not None:
                            tts_first_chunk_ms = (t_first_chunk - t_text_ready) * 1000
                            total_turn_ms = (t_first_chunk - t_speech_final) * 1000
                            
                            # Log individual stages at DEBUG level
                            logger.debug(f"[{state_ref['state']['call_sid']}] tts_first_chunk: {tts_first_chunk_ms:.0f}ms")
                            logger.debug(f"[{state_ref['state']['call_sid']}] total_turn: {total_turn_ms:.0f}ms")
                            
                            # Warnings check
                            is_rag = (intent == "answer_faq")
                            total_threshold = 3000.0 if is_rag else 1500.0
                            tts_threshold = 500.0
                            
                            warnings_list = []
                            if tts_first_chunk_ms > tts_threshold:
                                warnings_list.append(f"tts_start exceeded {tts_threshold}ms")
                            if total_turn_ms > total_threshold:
                                warnings_list.append(f"total exceeded {total_threshold}ms")
                                
                            warning_flag = " ⚠️  " if warnings_list else ""
                            warning_detail = f" ({', '.join(warnings_list)})" if warnings_list else ""
                            
                            logger.info(
                                f"[{state_ref['state']['call_sid']}] Turn {turn_num} latency summary:\n"
                                f"  graph: {graph_invocation_ms:.0f}ms (intent: {intent})\n"
                                f"  tts_start: {tts_first_chunk_ms:.0f}ms\n"
                                f"  total: {total_turn_ms:.0f}ms{warning_flag}{warning_detail}"
                            )
                            
                            RECENT_LATENCIES.append({
                                "call_sid": state_ref["state"]["call_sid"],
                                "turn": turn_num,
                                "intent": intent,
                                "speech_final_to_graph_start_ms": speech_final_to_graph_start_ms,
                                "graph_invocation_ms": graph_invocation_ms,
                                "tts_first_chunk_ms": tts_first_chunk_ms,
                                "total_turn_ms": total_turn_ms,
                                "timestamp": time.time()
                            })
                    
                    media_msg = encode_audio_for_twilio(stream_sid, chunk)
                    await websocket.send_text(media_msg)
                    if diagnostics:
                        diagnostics.log_tts_chunk_sent()
                    
    except Exception as e:
        logger.error(f"TTS streaming error: {e}")
        if diagnostics:
            diagnostics.log_error(f"TTS streaming error: {e}")
    finally:
        state_ref["state"]["is_speaking"] = False
        
        # Check for pending call transfer (escalation)
        if state_ref["state"].get("pending_transfer"):
            transfer_call_to_human(state_ref["state"]["call_sid"])

async def handle_barge_in(stream_sid: str, websocket: WebSocket, 
                           audio_queue: asyncio.Queue, state_ref: dict) -> None:
    """Called when speech_final fires while is_speaking == True."""
    state_ref["state"]["is_speaking"] = False
    clear_msg = build_clear_message(stream_sid)
    await websocket.send_text(clear_msg)
    # Drain the audio queue
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

async def silence_watchdog(
    speech_event: asyncio.Event,
    state_ref: dict,           # {"state": AgentState} mutable reference
    websocket: WebSocket,
    audio_queue: asyncio.Queue,
    tts_func: callable         # async function to generate TTS audio bytes
) -> None:
    """
    Monitor for caller silence and prompt or hang up as appropriate.
    
    Must be run as an asyncio Task. Cancel this task when the call ends
    to prevent resource leaks.
    
    Args:
        speech_event: Set by twilio_receiver on every Deepgram speech_final event.
                      This function clears it after waking.
        state_ref: Mutable dict wrapper {"state": AgentState} for cross-coroutine access
        websocket: Active Twilio WebSocket for sending clear/hangup signals
        audio_queue: TTS audio queue — watchdog prompts are enqueued here
        tts_func: Async callable that converts text to mulaw audio bytes
    """
    SILENCE_TIMEOUT = 5.0  # seconds
    logger.info("Silence watchdog task started.")
    
    while True:
        try:
            await asyncio.wait_for(speech_event.wait(), timeout=SILENCE_TIMEOUT)
            speech_event.clear()
            # Caller spoke — reset warning counter
            if state_ref["state"] is not None:
                state_ref["state"]["silence_warnings"] = 0
                
        except asyncio.TimeoutError:
            state = state_ref.get("state")
            if state is None:
                continue  # Call not fully initialized yet
            
            # If AI is currently speaking, do not count this as silence timeout
            if state.get("is_speaking", False):
                continue
            
            warnings = state.get("silence_warnings", 0)
            
            if warnings >= 2:
                # Second timeout — hang up
                logger.info("Silence watchdog: second timeout reached. Hanging up.")
                goodbye_audio = await tts_func(
                    "We haven't heard from you. Thank you for calling. Goodbye."
                )
                if goodbye_audio:
                    # Split into smaller audio chunks for smoother playback
                    for c in chunk_audio(goodbye_audio):
                        await audio_queue.put(c)
                await asyncio.sleep(3.0)  # Wait for goodbye to play
                await hangup_call(state["call_sid"])
                break
            else:
                # First timeout — prompt
                logger.info(f"Silence watchdog: first timeout reached. Warning count: {warnings + 1}")
                prompt_audio = await tts_func("Are you still there?")
                if prompt_audio:
                    for c in chunk_audio(prompt_audio):
                        await audio_queue.put(c)
                state["silence_warnings"] = warnings + 1
                
        except asyncio.CancelledError:
            logger.info("Silence watchdog task cancelled.")
            break  # Task cancelled — clean exit
        except Exception as e:
            logger.error(f"Error in silence_watchdog loop: {e}", exc_info=True)
            break

async def twilio_handler(websocket: WebSocket):
    """
    Main WebSocket orchestrator for Twilio Media Streams.
    Handles the lifecyle of a call, connection to Deepgram STT,
    and concurrent sender/receiver/watchdog execution.
    """
    global LAST_CALL_DIAGNOSTICS
    diagnostics = WebSocketDiagnostics(call_sid="pending")
    await websocket.accept()
    logger.info("Twilio WebSocket connection accepted.")

    audio_queue = asyncio.Queue()
    state = None
    stream_sid = None
    call_sid = None
    
    # Read caller_number from query parameters or default to Unknown
    caller_number = websocket.query_params.get("caller_number", "Unknown")

    # 1. Parse Twilio start event
    try:
        async for message in websocket:
            data = json.loads(message)
            event = data.get("event")
            if event == "connected":
                logger.info("Twilio connected event received.")
                continue
            elif event == "start":
                start_data = data["start"]
                stream_sid = start_data["streamSid"]
                call_sid = start_data["callSid"]
                diagnostics.call_sid = call_sid
                diagnostics.log_start_event()
                logger.info(f"Twilio start event. StreamSid: {stream_sid}, CallSid: {call_sid}")
                break
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected during startup.")
        diagnostics.log_error("WebSocket disconnected during startup.")
        # Store diagnostics of the failed startup
        LAST_CALL_DIAGNOSTICS = diagnostics.diagnose()
        return
    except Exception as e:
        logger.error(f"Error reading initial Twilio stream: {e}")
        diagnostics.log_error(f"Error reading initial Twilio stream: {e}")
        # Store diagnostics of the failed startup
        LAST_CALL_DIAGNOSTICS = diagnostics.diagnose()
        await websocket.close()
        return

    if not stream_sid or not call_sid:
        logger.error("Missing StreamSid or CallSid on startup.")
        diagnostics.log_error("Missing StreamSid or CallSid on startup.")
        # Store diagnostics of the failed startup
        LAST_CALL_DIAGNOSTICS = diagnostics.diagnose()
        await websocket.close()
        return

    # Initialize AgentState
    state = initial_state(call_sid=call_sid, stream_sid=stream_sid, caller_number=caller_number)
    state_ref = {"state": state}
    turn_count = 0

    # Initialize Deepgram client
    dg_client = AsyncDeepgramClient(api_key=settings.deepgram_api_key)

    # Event to notify watchdog of user speech
    speech_event = asyncio.Event()
    speech_event.set()

    # 2. Establish connection to Deepgram STT
    try:
        async with dg_client.listen.v1.connect(
            model="nova-3-medical",
            encoding="linear16",
            sample_rate=8000,
            endpointing=True
        ) as dg_socket:
            
            # Transcript accumulator for the turn
            accumulated_transcript = []



            async def generate_tts_bytes(text: str) -> bytes:
                """Generate raw mulaw audio bytes from text using Deepgram TTS."""
                try:
                    audio_client = dg_client.speak.v1.audio
                    audio_stream = audio_client.generate(
                        text=text,
                        model="aura-2-andromeda-en",
                        encoding="mulaw",
                        container="none",
                        sample_rate=8000
                    )
                    chunks = []
                    async for chunk in audio_stream:
                        chunks.append(chunk)
                    return b"".join(chunks)
                except Exception as e:
                    logger.error(f"Error generating TTS bytes in watchdog: {e}")
                    return b""

            async def run_agent_turn(user_text: str, t_speech_final: float):
                """Invokes LangGraph state machine and handles TTS output."""
                nonlocal state
                nonlocal turn_count
                turn_count += 1
                try:
                    t_graph_start = None
                    with LatencyTracker("speech_final_to_graph_start", call_sid=call_sid, warn_threshold_ms=100.0) as tracker1:
                        # Override start to measure from speech_final
                        tracker1.start = t_speech_final
                    
                    speech_final_to_graph_start_ms = tracker1.elapsed_ms
                    
                    with LatencyTracker("graph_invocation", call_sid=call_sid) as tracker2:
                        logger.info("Running LangGraph agent turn...")
                        updated_state, response_text = await run_agent_turn_graph(state, user_text)
                    
                    state = updated_state
                    state_ref["state"] = state
                    
                    graph_invocation_ms = tracker2.elapsed_ms
                    intent = state.get("current_intent", "unknown")
                    
                    is_rag = (intent == "answer_faq")
                    graph_threshold = 3000.0 if is_rag else 2000.0
                    if graph_invocation_ms > graph_threshold:
                        logger.warning(
                            f"[{call_sid}] graph_invocation: {graph_invocation_ms:.0f}ms (exceeded {graph_threshold}ms, intent: {intent})"
                        )
                    
                    if audio_queue.empty() and response_text:
                        await stream_response_to_caller(
                            response_text=response_text,
                            stream_sid=stream_sid,
                            websocket=websocket,
                            audio_queue=audio_queue,
                            state_ref=state_ref,
                            t_speech_final=t_speech_final,
                            speech_final_to_graph_start_ms=speech_final_to_graph_start_ms,
                            graph_invocation_ms=graph_invocation_ms,
                            turn_num=turn_count,
                            intent=intent,
                            diagnostics=diagnostics
                        )
                except Exception as e:
                    logger.error(f"Error running agent turn: {e}")

            async def on_message(message):
                """Callback invoked on receipt of Deepgram message."""
                if not isinstance(message, ListenV1Results):
                    return
                
                alternatives = message.channel.alternatives
                if not alternatives:
                    return
                
                transcript = alternatives[0].transcript.strip()
                if not transcript:
                    return
                
                # Barge-in: if user interrupts during AI output playback
                if state_ref["state"]["is_speaking"]:
                    logger.info("Barge-in triggered. Flushing outgoing TTS queue and Twilio buffer.")
                    await handle_barge_in(stream_sid, websocket, audio_queue, state_ref)

                accumulated_transcript.append(transcript)

                if message.speech_final:
                    final_text = " ".join(accumulated_transcript).strip()
                    accumulated_transcript.clear()
                    
                    if not final_text:
                        return
                    
                    diagnostics.log_speech_final(final_text)
                    logger.info(f"Caller Utterance completed: {final_text}")
                    state["silence_warnings"] = 0
                    speech_event.set()
                    
                    # Run the LangGraph agent turn with completion timestamp
                    t_speech_final = time.perf_counter()
                    asyncio.create_task(run_agent_turn(final_text, t_speech_final))

            # Register Deepgram message event handler
            dg_socket.on(EventType.MESSAGE, on_message)

            # Start Deepgram listening WebSocket reader task
            dg_listen_task = asyncio.create_task(dg_socket.start_listening())

            # Start Twilio sender task
            sender_task = asyncio.create_task(twilio_sender(websocket, audio_queue, state, diagnostics=diagnostics))

            # Start silence watchdog task
            watchdog_task = asyncio.create_task(
                silence_watchdog(speech_event, state_ref, websocket, audio_queue, generate_tts_bytes)
            )

            # Play welcome greeting on call start
            async def play_greeting():
                await asyncio.sleep(0.5)
                greeting = "Thank you for calling Saint Jude Hospital. How can I help you today?"
                state["conversation_history"].append(AIMessage(content=greeting))
                await stream_response_to_caller(greeting, stream_sid, websocket, audio_queue, state_ref, diagnostics=diagnostics)
            
            asyncio.create_task(play_greeting())

            # 3. Main Twilio receiver loop
            try:
                async for message in websocket:
                    data = json.loads(message)
                    event = data.get("event")
                    if event == "media":
                        payload = data["media"]["payload"]
                        pcm_bytes = decode_twilio_audio(payload)
                        diagnostics.log_audio_chunk()
                        await dg_socket.send_media(pcm_bytes)
                    elif event == "stop":
                        logger.info("Received Twilio stop event. Tearing down handler.")
                        break
            except WebSocketDisconnect:
                logger.info("Twilio WebSocket disconnected.")
                diagnostics.log_error("WebSocket disconnected mid-call")
            finally:
                # Cleanup tasks
                logger.info("Cancelling call handler subtasks...")
                watchdog_task.cancel()
                sender_task.cancel()
                dg_listen_task.cancel()
                
                # Wait for them to terminate cleanly
                await asyncio.gather(watchdog_task, sender_task, dg_listen_task, return_exceptions=True)

    except Exception as e:
        logger.error(f"Error in twilio_handler connection loop: {e}")
        diagnostics.log_error(f"Deepgram connection or loop failed: {e}")
        raise e
    finally:
        logger.info("Twilio handler session ended.")
        LAST_CALL_DIAGNOSTICS = diagnostics.diagnose()
        logger.info(diagnostics.summary())
