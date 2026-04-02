import os
from pymilvus import connections, utility
from dotenv import load_dotenv

load_dotenv()

MILVUS_HOST = "10.169.101.73"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "amazon_fashion_products"

connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

if utility.has_collection(MILVUS_COLLECTION):
    utility.drop_collection(MILVUS_COLLECTION)
    print(f"✓ Dropped collection: {MILVUS_COLLECTION}")
else:
    print(f"Collection {MILVUS_COLLECTION} doesn't exist")

print("✓ Ready to recreate with new schema")
