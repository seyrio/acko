import asyncio
import whisper
import torch
import numpy as np
import pyaudio
import warnings
import time
import redis  # synchronous Redis client
import redis.asyncio as redis_async  # async Redis client for FastAPI WebSocket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import threading

# ----------------------------
# Whisper Recorder with Redis
# ----------------------------
class FixedWhisperRecorder:
    def __init__(self, model_name="base", redis_client=None):
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
        self.redis_client = redis_client

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
            print("Transcribing with dtype fixes...")
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
        print("🎙️  Starting recording and transcription with dtype fixes")
        print("=" * 60)
        audio_array = self.record_audio_fixed(duration)
        if audio_array is None:
            print("❌ Recording failed")
            return None
        result = self.transcribe_with_dtype_fix(audio_array, language)
        if result and result.get('text', '').strip():
            text = result['text']
            print(f"✅ Success: '{text}'")
            if self.redis_client:
                try:
                    self.redis_client.xadd(
                        name="transcription_stream",
                        fields={
                            "text": text,
                            "timestamp": str(time.time())
                        },
                        maxlen=1000,
                        approximate=True
                    )
                    print("📤 Transcription sent to Redis stream.")
                except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
                    print(f"Redis connection error: {e}. Attempting to reconnect...")
                    try:
                        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                        self.redis_client.xadd(
                            name="transcription_stream",
                            fields={
                                "text": text,
                                "timestamp": str(time.time())
                            },
                            maxlen=1000,
                            approximate=True
                        )
                        print("📤 Transcription sent to Redis stream after reconnect.")
                    except Exception as e2:
                        print(f"Failed to push to Redis after reconnect: {e2}")
                except Exception as e:
                    print(f"Failed to push to Redis: {e}")
            return result
        else:
            print("❌ Transcription failed or returned empty text")
            return None

# ----------------------------
# FastAPI + WebSocket + Async Redis
# ----------------------------
app = FastAPI()

redis_client_async = redis_async.Redis(host='localhost', port=6379, db=0, decode_responses=True)

html = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Transcriptions</title>
</head>
<body>
    <h2>Live Redis Transcriptions</h2>
    <ul id="transcriptions"></ul>

    <script>
        const ws = new WebSocket("ws://localhost:8001/ws");
        const ul = document.getElementById("transcriptions");

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            const li = document.createElement("li");
            li.textContent = `[${new Date(parseFloat(data.timestamp) * 1000).toLocaleTimeString()}] ${data.text}`;
            ul.appendChild(li);
        };

        ws.onclose = () => {
            const li = document.createElement("li");
            li.textContent = "WebSocket connection closed.";
            ul.appendChild(li);
        };
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
    last_id = '0-0'
    try:
        while True:
            streams = await redis_client_async.xread({'transcription_stream': last_id}, block=5, count=1)
            if streams:
                for stream_name, messages in streams:
                    for message_id, message_data in messages:
                        await websocket.send_json(message_data)
                        last_id = message_id
            else:
                await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print("WebSocket disconnected")

# ----------------------------
# Run Whisper Recorder in background thread
# ----------------------------
def run_recorder():
    redis_client_sync = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

    try:
        redis_client_sync.ping()
        print("✅ Connected to Redis successfully!")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return

    recorder = FixedWhisperRecorder(model_name="base", redis_client=redis_client_sync)

    print("Press Ctrl+C to stop recording and transcribing.")
    while True:
        try:
            recorder.record_and_transcribe_fixed(duration=5, language="en")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Exception in recorder loop: {e}")
            time.sleep(1)

# ----------------------------
# Main entry: run FastAPI + recorder thread
# ----------------------------
import uvicorn

if __name__ == "__main__":
    recorder_thread = threading.Thread(target=run_recorder, daemon=True)
    recorder_thread.start()

    uvicorn.run("whisper_redis_fastapi:app", host="127.0.0.1", port=8001, reload=False)
