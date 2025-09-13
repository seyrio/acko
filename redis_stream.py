import redis

r = redis.Redis(host='localhost', port=6379, db=0)

# Get all messages from the stream
messages = r.xrange('transcription_stream', count=10)

for msg_id, fields in messages:
    text = fields[b'text'].decode()
    timestamp = fields[b'timestamp'].decode()
    print(f"[{timestamp}] {text}")
