import redis

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

try:
    if r.ping():
        print("✅ Redis is working!")
        r.flushdb()  # Clears current database (default is db 0)
        print("✅ Current database flushed!")
except redis.ConnectionError as e:
    print("❌ Redis not reachable:", e)
