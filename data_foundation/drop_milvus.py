from pymilvus import connections, utility

MILVUS_HOST = "10.169.101.75"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "amazon_fashion"

connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

if utility.has_collection(MILVUS_COLLECTION):
    utility.drop_collection(MILVUS_COLLECTION)
    print(f"✅ Dropped collection: {MILVUS_COLLECTION}")
else:
    print("ℹ️ Collection does not exist")
