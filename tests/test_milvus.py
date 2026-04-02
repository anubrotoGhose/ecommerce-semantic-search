from pymilvus import connections

# Replace with YOUR Milvus server's IP address
MILVUS_HOST = "10.169.101.73"   # e.g., "192.168.1.25" or public IP
MILVUS_PORT = "19530"

try:
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("✅ Successfully connected to Milvus at", MILVUS_HOST, MILVUS_PORT)
except Exception as e:
    print("❌ Failed to connect to Milvus:", e)