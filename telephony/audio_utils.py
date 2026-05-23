"""
Audio utilities for the ai_call_agent telephony pipeline.
Handles base64 encoding/decoding, audio format conversion (mulaw <-> linear16 PCM),
and queueing/chunking for Twilio media streams.
"""

import asyncio
import audioop
import base64
import json
from typing import List

def decode_twilio_audio(payload: str) -> bytes:
    """
    Decode base64-encoded mulaw (u-law) audio string from Twilio media event
    and convert it to raw PCM linear16 bytes ready for Deepgram STT.

    Args:
        payload: Base64 encoded mulaw audio payload string.

    Returns:
        Raw PCM linear16 bytes (16-bit, 8kHz, mono).
    """
    # Decode base64 to raw mulaw bytes
    mulaw_bytes = base64.b64decode(payload)
    # Convert mulaw to linear16 PCM (sample width = 2 bytes)
    pcm_bytes = audioop.ulaw2lin(mulaw_bytes, 2)
    return pcm_bytes

def encode_audio_for_twilio(stream_sid: str, mulaw_bytes: bytes) -> str:
    """
    Convert raw mulaw bytes from Deepgram TTS into a Twilio media JSON message string.

    Args:
        stream_sid: Twilio media stream SID.
        mulaw_bytes: Raw 8kHz mulaw bytes.

    Returns:
        JSON string ready to be sent over Twilio WebSocket.
    """
    payload = base64.b64encode(mulaw_bytes).decode("utf-8")
    message = {
        "event": "media",
        "streamSid": stream_sid,
        "media": {
            "payload": payload
        }
    }
    return json.dumps(message)

def build_clear_message(stream_sid: str) -> str:
    """
    Build a JSON message for Twilio to clear the audio buffer (stop playback/barge-in).

    Args:
        stream_sid: Twilio media stream SID.

    Returns:
        JSON string representing the clear event.
    """
    message = {
        "event": "clear",
        "streamSid": stream_sid
    }
    return json.dumps(message)

def create_audio_queue() -> asyncio.Queue:
    """
    Create a new asyncio.Queue for TTS audio chunks.

    Returns:
        An empty asyncio.Queue instance.
    """
    return asyncio.Queue()

def chunk_audio(audio_bytes: bytes, chunk_size: int = 8000) -> List[bytes]:
    """
    Split large raw audio bytes into smaller chunks for streaming to Twilio.

    Args:
        audio_bytes: Large buffer of raw audio.
        chunk_size: Size in bytes of each chunk. Default is 8000 bytes
                    (approx. 500ms at 8kHz 8-bit mulaw).

    Returns:
        List of audio byte chunks.
    """
    chunks = []
    for i in range(0, len(audio_bytes), chunk_size):
        chunks.append(audio_bytes[i:i + chunk_size])
    return chunks

if __name__ == "__main__":
    print("Running sanity checks for audio_utils.py...")
    
    # 1. Test build_clear_message
    clear_msg = build_clear_message("test_stream_sid")
    print(f"Clear Message: {clear_msg}")
    assert "clear" in clear_msg
    assert "test_stream_sid" in clear_msg
    
    # 2. Test create_audio_queue
    q = create_audio_queue()
    print(f"Queue created: {type(q)}")
    assert isinstance(q, asyncio.Queue)
    
    # 3. Test chunk_audio
    test_data = b"0123456789" * 10  # 100 bytes
    chunks = chunk_audio(test_data, chunk_size=30)
    print(f"Chunked 100 bytes into chunks of 30: {[len(c) for c in chunks]}")
    assert len(chunks) == 4
    assert len(chunks[0]) == 30
    assert len(chunks[3]) == 10
    
    # 4. Test encode_audio_for_twilio
    mulaw_test = b"\xff\x7f\x80"
    twilio_msg = encode_audio_for_twilio("sid123", mulaw_test)
    print(f"Twilio Media Message: {twilio_msg}")
    msg_dict = json.loads(twilio_msg)
    assert msg_dict["event"] == "media"
    assert msg_dict["streamSid"] == "sid123"
    assert base64.b64decode(msg_dict["media"]["payload"]) == mulaw_test

    # 5. Test decode_twilio_audio
    # Base64 for b"\xff\x7f\x80" is "/3+AgA==" or "/3+A" depending on padding/input
    payload_str = base64.b64encode(mulaw_test).decode("utf-8")
    pcm_out = decode_twilio_audio(payload_str)
    print(f"Decoded PCM bytes length: {len(pcm_out)}")
    # Each mulaw byte (8-bit) converts to linear16 PCM (16-bit = 2 bytes)
    assert len(pcm_out) == len(mulaw_test) * 2

    print("All sanity checks passed successfully!")
