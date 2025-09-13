#!/usr/bin/env python3

import whisper
import torch
import numpy as np
import pyaudio
import warnings
import tempfile
import wave
import os
import time
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import redis.asyncio as redis
import uvicorn
import threading

# ----------- Whisper Recorder Class -----------

class FixedWhisperRecorder:
    def __init__(self, model_name="base"):
        print(f"Loading Whisper model: {model_name}")
        
        torch.set_default_dtype(torch.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.model = whisper.load_model(model_name)
        self.model = self.model.float()
        if hasattr(self.model, 'encoder'):
            self.model.encoder = self.model.encoder.float()
        if hasattr(self.model, 'decoder'):
            self.model.decoder = self.model.decoder.float()

        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

    def record_audio_fixed(self, duration=5, sample_rate=16000):
        print(f"Recording for {duration} seconds at {sample_rate} Hz...")
        audio = pyaudio.PyAudio()
        device_index = None

        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                try:
                    test_stream = audio.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        input_device_index=i,
                        frames_per_buffer=1024
                    )
                    test_stream.close()
                    device_index = i
                    print(f"Using device: {info['name']}")
                    break
                except:
                    continue
        
        if device_index is None:
            print("Using default input device")

        stream_config = {
            'format': pyaudio.paInt16,
            'channels': 1,
            'rate': sample_rate,
            'input': True,
            'frames_per_buffer': 1024
        }
        if device_index is not None:
            stream_config['input_device_index'] = device_index
        
        stream = audio.open(**stream_config)
        frames = []
        frames_to_record = int(sample_rate / 1024 * duration)
        print("Recording... Speak now!")
        max_level = 0

        for i in range(frames_to_record):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
            chunk = np.frombuffer(data, dtype=np.int16)
            level = np.max(np.abs(chunk))
            max_level = max(max_level, level)
            if i % 10 == 0:
                progress = (i / frames_to_record) * 100
                print(f"\rProgress: {progress:.0f}% | Max level: {max_level}", end="", flush=True)

        print(f"\nRecording complete! Max level: {max_level}")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        if max_level < 100:
            print(f"⚠️  Audio level is very low ({max_level}). Try speaking louder.")

        audio_data = b''.join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_float32 = audio_array.astype(np.float32) / 32768.0
        return audio_float32

    def transcribe_with_dtype_fix(self, audio_array, language=None):
        if audio_array is None or len(audio_array) == 0:
            print("Empty audio array")
            return None
        
        try:
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            options = {
                'fp16': False,
                'language': language if language else None,
                'task': 'transcribe'
            }
            options = {k: v for k, v in options.items() if v is not None}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = self.model.transcribe(audio_array, **options)
            return result
            
        except Exception as e:
            print(f"Transcription failed: {e}")
            return None

    def record_and_transcribe_fixed(self, duration=5, language="en"):
        audio_array = self.record_audio_fixed(duration)
        if audio_array is None:
            print("❌ Recording failed")
            return None
        result = self.transcribe_with_dtype_fix(audio_array, language)
        if result and result.get('text', '').strip():
            text = result['text']
            print(f"✅ Success: '{text}'")
            self.redis_client.xadd(
                name="transcription_stream",
                fields={
                    "text": text,
                    "timestamp": str(time.time())
                }
            )
            print("📤 Transcription sent to Redis stream.")
            return result
        else:
            print("❌ Transcription failed or returned empty text")
            return None

# ----------- FastAPI Webserver -----------

app = FastAPI()
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
STREAM_KEY = "transcription_stream"

html = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Transcriptions from Redis Stream</title>
</head>
<body>
    <h1>Live Transcriptions from Redis Stream</h1>
    <ul id="messages"></ul>

    <script>
        let ws = new WebSocket("ws://" + location.host + "/ws");
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            const messages = document.getElementById("messages");
            const li = document.createElement("li");
            li.textContent = `${data.timestamp}: ${data.text}`;
            messages.appendChild(li);
        };
        ws.onopen = () => console.log("WebSocket connected");
        ws.onclose = () => console.log("WebSocket disconnected");
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_id = "0-0"  # Start from beginning, change to "$" for only new messages

    try:
        while True:
            streams = await redis_client.xread({STREAM_KEY: last_id}, block=0, count=10)
            if streams:
                for stream_key, messages in streams:
                    for message_id, fields in messages:
                        text = fields.get("text", "")
                        timestamp = fields.get("timestamp", "")
                        await websocket.send_json({"text": text, "timestamp": timestamp})
                        last_id = message_id
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# ----------- Run recorder in background thread -----------

def run_recorder():
    fix_pytorch_whisper_compatibility()
    recorder = FixedWhisperRecorder(model_name="base")
    try:
        while True:
            recorder.record_and_transcribe_fixed(duration=5, language="en")
            print("-" * 50)
    except KeyboardInterrupt:
        print("\n🛑 Recorder stopped by user.")

def fix_pytorch_whisper_compatibility():
    torch.set_default_dtype(torch.float32)
    os.environ['TORCH_DEFAULT_DTYPE'] = 'float32'
    os.environ['WHISPER_NO_FP16'] = '1'

# ----------- Main Entrypoint -----------

if __name__ == "__main__":
    # Start recorder in background thread
    recorder_thread = threading.Thread(target=run_recorder, daemon=True)
    recorder_thread.start()

    # Start FastAPI app with Uvicorn
    uvicorn.run("whisper_redis_fastapi:app", host="127.0.0.1", port=8001, reload=False)
